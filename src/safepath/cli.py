"""Command-line entry point: `python -m safepath <step>`."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from safepath.config import load_config
from safepath.data.clean import run_clean
from safepath.data.download import download_collisions
from safepath.data.report import run_report
from safepath.evaluate import run_evaluate
from safepath.features.build import run_features
from safepath.graph import run_graph
from safepath.locations import run_locations
from safepath.models.train import run_train
from safepath.risk import run_score
from safepath.routing import run_routing
from safepath.snap_report import run_snap_report
from safepath.snapping import run_snap

log = logging.getLogger("safepath")


def run_app(cfg: dict) -> None:
    """Launch the Streamlit app; blocks until the server stops."""
    script = Path(__file__).parent / "app" / "streamlit_app.py"
    result = subprocess.run([sys.executable, "-m", "streamlit", "run", str(script)], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"streamlit exited with code {result.returncode}")


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
    "score": ("Score 2026 risk and write it onto the routing graph", run_score),
    "route": ("Calibrate k and report the routing trade-off", run_routing),
    "app": ("Launch the Streamlit routing app", run_app),
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
    "score",
    "route",
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
        log.info("running step '%s'", step)
        try:
            STEPS[step][1](cfg)
        except Exception:
            # Stop the pipeline at the first failure; later steps must not run on stale inputs.
            log.exception("step '%s' failed; stopping", step)
            return 1
    return 0
