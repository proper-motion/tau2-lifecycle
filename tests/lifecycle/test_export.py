"""Phase 4: training export allowlist, private eval split, analytics sink."""

import json

import pytest
from test_campaign import TEMPLATES, _cfg, csat, scripted_runner

from tau2.lifecycle import export
from tau2.lifecycle.campaign import Variant, run_campaign
from tau2.lifecycle.kpis import window_labels
from tau2.lifecycle.world import open_ledger

SENTINEL = "SENTINEL_PRIVATE"


def _sentinel_runner():
    """Scripted agent whose private fields all carry the sentinel."""
    inner = scripted_runner()

    def run(task, variant, conv_id, intent_json):
        out = inner(task, variant, conv_id, intent_json)
        return out.model_copy(update={"task_id": f"{SENTINEL}_task|{out.task_id}"})

    return run


@pytest.fixture
def ledger(tmp_path):
    # Variant names stay "good"/"bad" (the scripted agent keys off them); their
    # absence from training rows is checked by name in the leakage test.
    cfg = _cfg(
        days=40,
        variants=[
            Variant(name="good", agent_llm="x"),
            Variant(name="bad", agent_llm="x", failure_rate=0.3),
        ],
    )
    run_campaign(cfg, tmp_path, _sentinel_runner(), csat, TEMPLATES)
    conn = open_ledger(tmp_path / "ledger.sqlite")
    # Plant the sentinel in every private store the export must never read.
    with conn:
        conn.execute(
            "UPDATE oracle SET payload = json_set(payload, '$.sentinel', ?)",
            (SENTINEL,),
        )
        conn.execute(
            "UPDATE events SET payload = json_set(payload, '$.intent_note', ?,"
            " '$.accepts_offer', ?) WHERE type IN ('conversation', 'contact_scheduled')",
            (SENTINEL, SENTINEL),
        )
    return conn, tmp_path


def test_training_rows_never_contain_private_data(ledger):
    conn, _ = ledger
    rows = export.training_rows(conn, cutoff_day=39)
    assert rows
    blob = json.dumps(rows)
    assert SENTINEL not in blob
    for forbidden in (
        "good",
        "bad",
        "mobile_data_issue",
        "mms_issue",
        "task_success",
        "base_churn",
    ):
        assert forbidden not in json.dumps(
            [{k: v for k, v in r.items() if k != "transcript"} for r in rows]
        )
    assert all(tuple(r) == export.TRAIN_FIELDS for r in rows)


def test_analytics_payloads_never_contain_private_data(ledger):
    conn, _ = ledger
    events = export.analytics_events(conn, cutoff_day=39)
    assert events
    for payload in (export.to_mixpanel(events), export.to_amplitude(events)):
        assert SENTINEL not in json.dumps(payload, default=str)


def test_only_mature_non_infra_conversations_are_exported(ledger):
    conn, _ = ledger
    rows = export.training_rows(conn, cutoff_day=39)
    labels = window_labels(conn, 39)
    mature = [r for r in labels if r["churn_or_downgrade_28d"] is not None]
    assert len(rows) == len(mature) < len(labels)  # recent windows still open
    assert all(
        r["virtual_day"] + 28 <= 39 or r["churn_or_downgrade_28d"] == 1 for r in rows
    )


def test_customer_key_is_shared_across_variants_and_private_file_is_separate(ledger):
    conn, path = ledger
    counts = export.write_exports(conn, path / "export", cutoff_day=39)
    train = [
        json.loads(x)
        for x in (path / "export/train_rows.jsonl").read_text().splitlines()
    ]
    private = [
        json.loads(x)
        for x in (path / "export/private/eval_private.jsonl").read_text().splitlines()
    ]
    assert counts["train_rows"] == len(train) and counts["private_rows"] == len(private)
    by_row = {p["row_id"]: p for p in private}
    assert all(r["row_id"] in by_row for r in train)
    keys = {}
    for r in train:
        keys.setdefault(r["customer_key"], set()).add(by_row[r["row_id"]]["variant"])
    assert any(len(v) == 2 for v in keys.values())  # same customer, both variants
    assert any(p["injected_failures"] > 0 or p["variant"] == "good" for p in private)


def test_analytics_ids_and_timestamps_are_stable_and_virtual(ledger):
    conn, _ = ledger
    a, b = export.analytics_events(conn, 39), export.analytics_events(conn, 39)
    assert [e["insert_id"] for e in a] == [e["insert_id"] for e in b]
    assert len({e["insert_id"] for e in a}) == len(a)
    day0 = [e for e in a if e["properties"]["virtual_day"] == 0][0]
    assert day0["time"] == export.VIRTUAL_EPOCH
    assert all(e["properties"]["synthetic"] is True for e in a)


class _Resp:
    def __init__(self, code, text=""):
        self.status_code, self.text = code, text


def test_sender_retries_rate_limits_then_succeeds(ledger, monkeypatch):
    conn, _ = ledger
    events = export.analytics_events(conn, 39)
    n_before = conn.execute("SELECT count(*) FROM events").fetchone()[0]
    codes = iter([429, 503, 200] + [200] * 100)
    calls = []
    monkeypatch.setattr(export.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        export.requests, "post", lambda *a, **k: calls.append(k) or _Resp(next(codes))
    )
    assert export.send_amplitude(events, "key") == len(events)
    assert len(calls) == 2 + -(-len(events) // 1000)
    assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == n_before


def test_sender_raises_on_client_error(ledger, monkeypatch):
    conn, _ = ledger
    events = export.analytics_events(conn, 39)
    monkeypatch.setattr(
        export.requests, "post", lambda *a, **k: _Resp(400, "bad project")
    )
    with pytest.raises(RuntimeError, match="HTTP 400 bad project"):
        export.send_mixpanel(events, "123", "u", "s")
