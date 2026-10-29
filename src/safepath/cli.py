"""Command-line entry point: `python -m safepath <step>`."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

from safepath.config import load_config
from safepath.data.download import download_collisions


def _not_implemented(step: str) -> Callable[[dict], None]:
    def run(cfg: dict) -> None:
        raise SystemExit(f"step '{step}' is not implemented yet")

    return run


STEPS: dict[str, tuple[str, Callable[[dict], None]]] = {
    "download": ("Download Toronto collision records", download_collisions),
    "report-data": ("Write the collision data report", _not_implemented("report-data")),
    "graph": ("Build the OSM road graph", _not_implemented("graph")),
    "snap": ("Snap collisions to road segments", _not_implemented("snap")),
    "features": ("Build segment-year features and labels", _not_implemented("features")),
    "train": ("Train baselines and the risk model", _not_implemented("train")),
    "evaluate": ("Evaluate models on val and test years", _not_implemented("evaluate")),
    "app": ("Launch the Streamlit routing app", _not_implemented("app")),
}
PIPELINE = ["download", "graph", "snap", "features", "train", "evaluate"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="safepath", description=__doc__)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    sub = parser.add_subparsers(dest="step", required=True, metavar="<step>")
    for name, (help_text, _) in STEPS.items():
        sub.add_parser(name, help=help_text)
    sub.add_parser("all", help="Run the full pipeline: " + " -> ".join(PIPELINE))
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    steps = PIPELINE if args.step == "all" else [args.step]
    for step in steps:
        STEPS[step][1](cfg)
    return 0
