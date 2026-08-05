"""Command line entry point.

Three verbs. ``compute`` builds a factor from a JSON definition and writes the series, the
chart and a run record next to each other. ``fields`` and ``schemes`` print the filter
vocabulary and the weighting options, which is faster than reading the source.

    semantic-factors compute --spec crop_infestation.json --out ./crop
    semantic-factors fields --search materiality
    semantic-factors schemes
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from semantic_factors import filters as filter_rules
from semantic_factors import logs
from semantic_factors.errors import SemanticFactorError
from semantic_factors.factor import SemanticFactor
from semantic_factors.version import __version__
from semantic_factors.weights import WEIGHT_SCHEMES

#: Keys a spec file may carry that describe the factor rather than how to compute it.
DEFINITION_KEYS = (
    "anchors", "poles", "pole_hi", "pole_lo", "filters", "name", "floor", "aggregate",
    "tanh_scale"
)

#: Keys a spec file may carry that are passed straight to compute().
COMPUTE_KEYS = ("weights", "residualize", "stabilize", "start_from", "stop_at", "api_key",
                "base_url", "embed_api_key", "embed_base_url", "cache_dir", "top_events",
                "per_date_limit", "polarity_weighted")


def main(argv: Optional[List[str]] = None) -> int:
    """
    Run the command line interface.

    :param argv: Argument list, or None to read sys.argv.
    :return: Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "compute":
            return run_compute(args=args)
        if args.command == "fields":
            return run_fields(search=args.search)
        if args.command == "schemes":
            return run_schemes()
    except SemanticFactorError as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return 1
    parser.print_help()
    return 2


def run_compute(args: argparse.Namespace) -> int:
    """
    Build one factor from a spec file and write its outputs.

    :param args: Parsed arguments carrying spec, out and quiet.
    :return: Process exit code.
    """
    logs.set_verbose(enabled=not args.quiet)
    with open(args.spec, encoding="utf-8") as handle:
        spec: Dict[str, Any] = json.load(handle)
    unknown = set(spec) - set(DEFINITION_KEYS) - set(COMPUTE_KEYS)
    if unknown:
        raise SemanticFactorError(
            f"unknown spec keys: {', '.join(sorted(unknown))}. "
            f"Known keys are {', '.join(sorted(DEFINITION_KEYS + COMPUTE_KEYS))}."
        )
    factor = SemanticFactor(**{key: spec[key] for key in DEFINITION_KEYS if key in spec})
    factor.compute(**{key: spec[key] for key in COMPUTE_KEYS if key in spec})

    os.makedirs(name=args.out, exist_ok=True)
    stem = safe_output_stem(output_dir=args.out, factor_name=factor.name)
    factor.to_csv(path=f"{stem}_daily.csv")
    factor.monthly().write_csv(file=f"{stem}_monthly.csv")
    with open(f"{stem}_run.json", "w", encoding="utf-8") as handle:
        json.dump(factor.diagnostics, handle, indent=2, sort_keys=True)
    if factor.events is not None:
        factor.events.write_csv(file=f"{stem}_events.csv")
    if not args.no_plot:
        write_chart(
            factor=factor,
            path=f"{stem}.png"
        )
    print(f"wrote {stem}_daily.csv and its companions", flush=True)
    return 0


def safe_output_stem(output_dir: str, factor_name: str) -> str:
    """Keep a factor-controlled filename inside the caller's output directory."""
    name = str(factor_name).strip()
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or Path(name).name != name
    ):
        raise SemanticFactorError(
            "factor name must be a single filename, without an absolute path or parent traversal"
        )
    output = Path(output_dir).resolve()
    candidate = (output / name).resolve()
    try:
        candidate.relative_to(output)
    except ValueError as error:
        raise SemanticFactorError("factor output path escaped --out") from error
    return str(candidate)


def write_chart(factor: SemanticFactor, path: str) -> None:
    """
    Draw the factor, reporting rather than failing when matplotlib is absent.

    :param factor: The computed factor.
    :param path: PNG path to write.
    :return: None.
    """
    try:
        factor.plot(annotate=True, path=path)
    except ImportError:
        logs.warn(message="matplotlib is not installed, so no chart was written")


def run_fields(search: Optional[str]) -> int:
    """
    Print the filterable fields, optionally narrowed to a search term.

    :param search: Substring to match, or None for everything.
    :return: Process exit code.
    """
    reverse: Dict[str, List[str]] = {}
    for alias, field in filter_rules.FIELD_ALIASES.items():
        reverse.setdefault(field, []).append(alias)
    for field, kind in sorted(filter_rules.FIELD_TYPES.items()):
        aliases = ", ".join(sorted(reverse.get(field, [])))
        if search and search.lower() not in f"{field} {aliases}".lower():
            continue
        suffix = f"  (also: {aliases})" if aliases else ""
        print(f"{field:<36} {kind:<11}{suffix}", flush=True)
    return 0


def run_schemes() -> int:
    """
    Print the weighting schemes and what each one measures.

    :return: Process exit code.
    """
    for name, description in WEIGHT_SCHEMES.items():
        print(f"{name:<14} {description}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser.

    :return: Configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="semantic-factors",
        description="Build quantitative factors from NOSIBLE World."
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command")

    compute = commands.add_parser("compute", help="build a factor from a JSON definition")
    compute.add_argument("--spec", required=True, help="path to the factor definition")
    compute.add_argument("--out", required=True, help="directory to write outputs into")
    compute.add_argument("--no-plot", action="store_true", help="skip the chart")
    compute.add_argument("--quiet", action="store_true", help="suppress progress output")

    fields = commands.add_parser("fields", help="list the filterable event fields")
    fields.add_argument("--search", default=None, help="only show fields matching this text")

    commands.add_parser("schemes", help="list the weighting schemes")
    return parser


if __name__ == "__main__":
    sys.exit(main())
