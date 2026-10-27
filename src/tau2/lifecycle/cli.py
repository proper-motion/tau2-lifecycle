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

    export = commands.add_parser(
        "export", help="Write training rows and private eval data for the ORM"
    )
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--cutoff", type=int, required=True)
    export.add_argument("--dest", type=Path, help="Defaults to <out>/export")
    export.set_defaults(func=_export)

    analytics = commands.add_parser(
        "analytics", help="Send business events to a synthetic analytics project"
    )
    analytics.add_argument("--out", type=Path, required=True)
    analytics.add_argument("--cutoff", type=int, required=True)
    analytics.add_argument("--to", choices=["mixpanel", "amplitude"], required=True)
    analytics.add_argument(
        "--project", help="Mixpanel project id of the synthetic project (no default)"
    )
    analytics.add_argument(
        "--dry-run", action="store_true", help="Print payload counts, send nothing"
    )
    analytics.set_defaults(func=_analytics)


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


def _export(args) -> None:
    from tau2.lifecycle.export import write_exports
    from tau2.lifecycle.world import open_ledger

    conn = open_ledger(args.out / "ledger.sqlite")
    counts = write_exports(conn, args.dest or args.out / "export", args.cutoff)
    print(json.dumps(counts, indent=2))


def _analytics(args) -> None:
    import os

    from tau2.lifecycle import export
    from tau2.lifecycle.world import open_ledger

    events = export.analytics_events(
        open_ledger(args.out / "ledger.sqlite"), args.cutoff
    )
    if args.dry_run:
        print(json.dumps({"events": len(events), "to": args.to}, indent=2))
        return
    if args.to == "mixpanel":
        if not args.project:
            raise SystemExit("--project is required: send only to a synthetic project")
        sent = export.send_mixpanel(
            events,
            args.project,
            os.environ["MIXPANEL_SERVICE_ACCOUNT_USERNAME"],
            os.environ["MIXPANEL_SERVICE_ACCOUNT_SECRET"],
        )
    else:
        sent = export.send_amplitude(events, os.environ["AMPLITUDE_API_KEY"])
    print(json.dumps({"sent": sent, "to": args.to}, indent=2))
