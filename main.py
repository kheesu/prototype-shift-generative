import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from generate.prototypes import generate_prototypes, generate_prototypes_persona, save_prototypes
from generate.usages import generate_usages, generate_usages_persona, save_usages

OUTPUTS_DIR = Path("outputs")
USAGES_DIR = OUTPUTS_DIR / "usages"
PROTOTYPES_DIR = OUTPUTS_DIR / "prototypes"

DEFAULT_TEXT_MODEL = "gpt-4o"
DEFAULT_N_USAGES = 10
DEFAULT_N_PROTOTYPES = 4


def run(args: argparse.Namespace) -> None:
    n_usages = args.n if args.n is not None else DEFAULT_N_USAGES
    n_prototypes = args.n if args.n is not None else DEFAULT_N_PROTOTYPES
    text_model = args.tm or DEFAULT_TEXT_MODEL
    vllm_url = args.vllm_url

    time_period = args.time_period

    for word in args.targets:
        if not args.prototypes_only:
            print(f"[usages] {word} ...")
            if time_period:
                usages = generate_usages_persona(word, time_period, n=n_usages, model=text_model, base_url=vllm_url)
            else:
                usages = generate_usages(word, n=n_usages, model=text_model, base_url=vllm_url)
            path = save_usages(word, usages, model=text_model, output_dir=USAGES_DIR)
            print(f"  -> {path} ({len(usages)} usages)")

        if not args.usages_only:
            print(f"[prototypes] {word} ...")
            if time_period:
                prototypes = generate_prototypes_persona(word, time_period, n=n_prototypes, model=text_model, base_url=vllm_url)
            else:
                prototypes = generate_prototypes(word, n=n_prototypes, model=text_model, base_url=vllm_url)
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
    parser.add_argument(
        "--time-period", default=None, metavar="ERA",
        help='Persona time period for generation (e.g. "1920s America", "Medieval Europe")',
    )
    parser.add_argument(
        "--vllm-url", default=None, metavar="URL",
        help="Base URL of a vLLM OpenAI-compatible server (e.g. http://localhost:8000/v1)",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
