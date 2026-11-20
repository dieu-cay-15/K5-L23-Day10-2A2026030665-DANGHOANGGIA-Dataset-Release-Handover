"""Run the available validator rules from the command line."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validator.detection_rules import validate_detection_rules
from validator.rules_metadata import check_metadata
from validator.rules_panoptic import check_panoptic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("delivery", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--candidates", type=Path)
    args = parser.parse_args()
    if args.candidates is not None:
        parser.error("--candidates requires the group's cross-check module")

    errors = (
        validate_detection_rules(args.delivery)
        + check_metadata(args.delivery)
        + check_panoptic(args.delivery)
    )
    errors.sort(key=lambda item: (item["type"], item["target"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"errors": errors}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
