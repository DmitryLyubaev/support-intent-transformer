import argparse
import json
from pathlib import Path

from p24.study import CapacityWait, StudyNotReady, run_stage, run_worker


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Explicit bounded support-intent study stages")
    parser.add_argument(
        "stage",
        choices=[
            "acquire",
            "prepare",
            "baseline",
            "feasibility",
            "train",
            "evaluate",
            "readback",
            "serve-check",
            "report",
            "enrich",
            "_worker",
        ],
    )
    parser.add_argument("--config", type=Path, default=Path("config/study.json"))
    parser.add_argument("--root", type=Path, default=Path(".local/study"))
    parser.add_argument("--purpose", choices=["feasibility", "measured"])
    args = parser.parse_args(argv)
    try:
        if args.stage == "_worker":
            run_worker(args.config, args.root, args.purpose)
            return 0
        result = run_stage(args.stage, args.config, args.root)
        print(json.dumps(result, ensure_ascii=False, default=str))
        return 3 if result.get("status") == "review_required" else 0
    except CapacityWait as exc:
        print(json.dumps({"status": "capacity_wait", "detail": str(exc)}))
        return 3
    except (StudyNotReady, FileExistsError) as exc:
        print(json.dumps({"status": "refused", "detail": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
