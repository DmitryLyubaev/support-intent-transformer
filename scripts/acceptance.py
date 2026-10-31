import argparse
import json
from pathlib import Path

from p24.study import verify_acceptance

parser = argparse.ArgumentParser(
    description="Verify retained A1-A8 evidence without refitting/scoring"
)
parser.add_argument("--root", type=Path, default=Path(".local/study"))
args = parser.parse_args()
result = verify_acceptance(args.root)
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["complete"] else 1)
