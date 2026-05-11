import argparse
from pathlib import Path

from generate.prototypes import generate_prototypes, save_prototypes
from generate.usages import generate_usages, save_usages

OUTPUTS_DIR = Path("outputs")
USAGES_DIR = OUTPUTS_DIR / "usages"
PROTOTYPES_DIR = OUTPUTS_DIR / "prototypes"

DEFAULT_TEXT_MODEL = "claude-sonnet-4-6"
DEFAULT_N_USAGES = 10
DEFAULT_N_PROTOTYPES = 4


def run(args: argparse.Namespace) -> None:
    n_usages = args.n if args.n is not None else DEFAULT_N_USAGES
    n_prototypes = args.n if args.n is not None else DEFAULT_N_PROTOTYPES
    text_model = args.tm or DEFAULT_TEXT_MODEL

    for word in args.targets:
        if not args.prototypes_only:
            print(f"[usages] {word} ...")
            usages = generate_usages(word, n=n_usages, model=text_model)
            path = save_usages(word, usages, model=text_model, output_dir=USAGES_DIR)
            print(f"  -> {path} ({len(usages)} usages)")

        if not args.usages_only:
            print(f"[prototypes] {word} ...")
            prototypes = generate_prototypes(word, n=n_prototypes, model=text_model)
            path = save_prototypes(word, prototypes, model=text_model, output_dir=PROTOTYPES_DIR)
            print(f"  -> {path} ({len(prototypes)} prototypes)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate prototypes and example usages for target words."
    )
    parser.add_argument(
        "-t", "--targets", nargs="+", required=True, metavar="WORD",
        help="One or more target words",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--usages-only", action="store_true", help="Skip prototype generation")
    mode.add_argument("--prototypes-only", action="store_true", help="Skip usage generation")
    parser.add_argument(
        "--n", type=int, default=None,
        help="Number of outputs per word (default: 10 for usages, 4 for prototypes)",
    )
    parser.add_argument("--tm", default=None, metavar="MODEL", help="Text model override")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
