"""`tau2 lifecycle ...` commands: run, resume, cancel and inspect campaigns."""

import argparse
import json
from pathlib import Path

import yaml


def add_lifecycle_args(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="lifecycle_command", required=True)

    run = commands.add_parser("run", help="Start or resume a campaign")
    run.add_argument("--config", type=Path, required=True, help="Campaign YAML/JSON")
    run.add_argument("--out", type=Path, required=True, help="Campaign directory")
    run.set_defaults(func=_run)

    cancel = commands.add_parser(
        "cancel", help="Stop a running campaign at the next day boundary"
    )
    cancel.add_argument("--out", type=Path, required=True)
    cancel.set_defaults(func=_cancel)

    kpis = commands.add_parser("kpis", help="Print per-variant KPIs from the ledger")
    kpis.add_argument("--out", type=Path, required=True)
    kpis.add_argument("--cutoff", type=int, required=True, help="Virtual day cutoff")
    kpis.set_defaults(func=_kpis)


def _run(args) -> None:
    from tau2.lifecycle.campaign import CampaignConfig, run_campaign

    raw = yaml.safe_load(args.config.read_text())
    (args.out / "CANCEL").unlink(missing_ok=True)
    status = run_campaign(CampaignConfig.model_validate(raw), args.out)
    print(json.dumps(status, indent=2))


def _cancel(args) -> None:
    (args.out / "CANCEL").touch()
    print(f"Cancel requested; {args.out} stops at the next day boundary.")


def _kpis(args) -> None:
    from tau2.lifecycle.kpis import cohort_kpis
    from tau2.lifecycle.world import open_ledger

    conn = open_ledger(args.out / "ledger.sqlite")
    print(json.dumps(cohort_kpis(conn, args.cutoff), indent=2))
