"""CLI entry points for the paper's generated-response numerical studies."""

import argparse
import sys
from pathlib import Path

from . import followup60, main20, supplement20, verify


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m legacy",
        description="Run the archived PCOC generated-response protocols with synthetic objects.",
    )
    commands = parser.add_subparsers(dest="study", required=True)
    for name, description in (
        ("main20", "Generate the 20-round main benchmark"),
        (
            "supplement20",
            "Generate the supplemental 20-round scenarios and analytic costs",
        ),
        ("followup60", "Generate the matched 60-round recovery follow-up"),
    ):
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument(
            "--output", type=Path, required=True, help="New directory for output files"
        )
        if name != "main20":
            command.add_argument(
                "--main",
                type=Path,
                required=True,
                help="Completed main20 output directory",
            )
    check_parser = commands.add_parser(
        "verify",
        help="Compare generated NPZ statistics with the archived author arrays",
    )
    check_parser.add_argument(
        "--run", type=Path, required=True, help="Completed generated-response output"
    )
    check_parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Original data folder containing the NPZ archives",
    )
    args = parser.parse_args(argv)
    if args.study == "verify":
        try:
            verify.run(args.run, args.reference)
        except (OSError, RuntimeError, KeyError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 1
        return 0
    if args.output.exists():
        parser.error(f"Output directory already exists: {args.output}")
    if args.study != "main20":
        from json import JSONDecodeError, loads

        try:
            manifest = loads((args.main / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, JSONDecodeError) as error:
            parser.error(f"Could not read main20 manifest at {args.main}: {error}")
        if manifest.get("study") != "main20":
            parser.error(f"Expected a completed main20 output at {args.main}")
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        if args.study == "main20":
            main20.run(args.output)
        elif args.study == "supplement20":
            supplement20.run(args.output, args.main)
        else:
            followup60.run(args.output, args.main)
    except (OSError, RuntimeError, ValueError) as error:
        print(
            f"{error}\nIncomplete outputs, if any, are in {args.output}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
