"""Command-line entry point: `python -m safepath <step>`."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable

from safepath.config import load_config
from safepath.data.clean import run_clean
from safepath.data.download import download_collisions
from safepath.data.report import run_report
from safepath.evaluate import run_evaluate
from safepath.features.build import run_features
from safepath.graph import run_graph
from safepath.locations import run_locations
from safepath.models.train import run_train
from safepath.snap_report import run_snap_report
from safepath.snapping import run_snap


def _not_implemented(step: str) -> Callable[[dict], None]:
    def run(cfg: dict) -> None:
        raise SystemExit(f"step '{step}' is not implemented yet")

    return run


STEPS: dict[str, tuple[str, Callable[[dict], None]]] = {
    "download": ("Download Toronto collision records", download_collisions),
    "clean": ("Clean, deduplicate, and filter collisions", run_clean),
    "report-data": ("Write the collision data report", run_report),
    "graph": ("Build the OSM road graph and consolidated intersections", run_graph),
    "locations": ("Analyse crash points against intersections, roads, and parking", run_locations),
    "snap": ("Assign collisions to intersections or mid-block segments", run_snap),
    "report-snap": ("Write the snapping report", run_snap_report),
    "features": ("Build intersection-year labels and features", run_features),
    "train": ("Fit baselines and LightGBM risk models", run_train),
    "evaluate": ("Evaluate models on val 2024 and test 2025", run_evaluate),
    "app": ("Launch the Streamlit routing app", _not_implemented("app")),
}
PIPELINE = [
    "download",
    "clean",
    "report-data",
    "graph",
    "locations",
    "snap",
    "report-snap",
    "features",
    "train",
    "evaluate",
]


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
