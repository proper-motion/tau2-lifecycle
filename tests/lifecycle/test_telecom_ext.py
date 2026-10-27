import itertools

import pytest

from tau2.domains.telecom.data_model import AccountStatus, LineStatus, TelecomDB
from tau2.domains.telecom.environment import get_environment as upstream_env
from tau2.domains.telecom.utils import TELECOM_DB_PATH
from tau2.lifecycle.telecom_ext import (
    ANNUAL_CREDIT_CAP,
    POLICY_ADDENDUM,
    RETENTION_CREDIT,
    LifecycleTelecomTools,
    get_environment,
    make_intent_task,
    retention_eligible,
)
from tau2.lifecycle.world import (
    TEMPLATE_NAME,
    check_applicable,
    instantiate_task,
    make_customers,
)
from tau2.registry import registry

NEW_TOOLS = {"change_plan", "cancel_line", "apply_goodwill_credit"}


@pytest.fixture
def tools():
    return LifecycleTelecomTools(TelecomDB.load(TELECOM_DB_PATH))


def _state(**overrides):
    c = make_customers(1, seed=1)[0]
    return {
        **c.model_dump(exclude={"traits"}),
        "plan_id": "P1002",
        "account_status": "Active",
        "goodwill_credit_used_this_year": 25.0,
        **overrides,
    }


# ---------------------------------------------------------------- tools


def test_change_plan(tools):
    out = tools.change_plan("C1001", "L1002", "P1001")
    assert tools._get_line_by_id("L1002").plan_id == "P1001"
    assert out["old_price_per_month"] == 65.0 and out["new_price_per_month"] == 40.0
    with pytest.raises(ValueError, match="already on plan"):
        tools.change_plan("C1001", "L1002", "P1001")
    with pytest.raises(ValueError, match="not found"):
        tools.change_plan("C1001", "L1002", "P9999")
    tools.suspend_line("C1001", "L1002", "test")
    with pytest.raises(ValueError, match="active"):
        tools.change_plan("C1001", "L1002", "P1003")


def test_cancel_line_closes_account_only_when_last_line_closes(tools):
    customer = tools.get_customer_by_id("C1001")
    tools.cancel_line("C1001", "L1002", "too expensive")
    assert tools._get_line_by_id("L1002").status == LineStatus.CLOSED
    assert customer.account_status == AccountStatus.ACTIVE
    with pytest.raises(ValueError, match="already closed"):
        tools.cancel_line("C1001", "L1002", "again")
    for lid in customer.line_ids:
        if tools._get_line_by_id(lid).status != LineStatus.CLOSED:
            tools.cancel_line("C1001", lid, "leaving")
    assert customer.account_status == AccountStatus.CLOSED


def test_goodwill_credit_respects_annual_cap(tools):
    customer = tools.get_customer_by_id("C1001")
    assert customer.goodwill_credit_used_this_year == 25.0  # template pre-used
    tools.apply_goodwill_credit("C1001", RETENTION_CREDIT)
    assert customer.goodwill_credit_used_this_year == 45.0
    with pytest.raises(ValueError, match="cap"):
        tools.apply_goodwill_credit("C1001", RETENTION_CREDIT)
    with pytest.raises(ValueError, match="positive"):
        tools.apply_goodwill_credit("C1001", -5)
    assert customer.goodwill_credit_used_this_year <= ANNUAL_CREDIT_CAP


# ---------------------------------------------------------------- domain isolation (D13/OV9)


def test_upstream_telecom_is_unchanged_and_lifecycle_domain_adds_tools():
    up, lc = upstream_env(), registry.get_env_constructor("telecom_lifecycle")()
    up_tools = {t.name for t in up.get_tools()}
    lc_tools = {t.name for t in lc.get_tools()}
    assert len(up_tools) == 13 and not (NEW_TOOLS & up_tools)
    assert lc_tools == up_tools | NEW_TOOLS
    assert "## Cancel Line" not in up.policy
    assert POLICY_ADDENDUM.strip() in lc.policy


# ---------------------------------------------------------------- intent tasks


@pytest.mark.parametrize(
    "intent,accepts",
    itertools.product(["downgrade", "upgrade", "cancel"], [True, False]),
)
def test_intent_tasks_have_a_valid_gold_path(intent, accepts):
    state = _state()
    task = instantiate_task(make_intent_task(state, intent, accepts), state)
    assert TEMPLATE_NAME not in task.model_dump_json()
    assert check_applicable(task, env_factory=get_environment) is None


def test_cancel_gold_path_follows_eligibility():
    eligible, capped = _state(), _state(goodwill_credit_used_this_year=40.0)
    assert retention_eligible(eligible) and not retention_eligible(capped)

    def gold(s, a):
        task = make_intent_task(s, "cancel", a)
        return [x.name for x in task.evaluation_criteria.actions]

    assert gold(eligible, True) == ["apply_goodwill_credit"]
    assert gold(eligible, False) == ["cancel_line"]
    assert gold(capped, True) == ["cancel_line"]  # no offer allowed (OV3)
    task = instantiate_task(make_intent_task(capped, "cancel", True), capped)
    assert check_applicable(task, env_factory=get_environment) is None


def test_intent_task_ids_are_deterministic():
    s = _state(persona="Hard")
    a = make_intent_task(s, "cancel", True)
    assert a == make_intent_task(s, "cancel", True)
    assert a.id == "[lifecycle]cancel_accept[PERSONA:Hard]"
