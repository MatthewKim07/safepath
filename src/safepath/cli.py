"""Command-line entry point: `python -m safepath <step>`."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from safepath.config import load_config, ottawa_config, toronto_config
from safepath.data.clean import run_clean
from safepath.data.download import download_collisions
from safepath.data.mto import run_mto
from safepath.data.ottawa import run_ottawa_data
from safepath.data.report import run_report
from safepath.data.volume_report import run_volumes
from safepath.evaluate import run_evaluate
from safepath.explain import run_explain
from safepath.exposure import run_exposure
from safepath.features.build import run_features
from safepath.graph import run_graph
from safepath.locations import run_locations
from safepath.models.calibration import run_calibration
from safepath.models.eb import run_eb
from safepath.models.spf import run_spf
from safepath.models.train import run_train
from safepath.models.volume_report import run_volume_model
from safepath.ottawa.build import run_ottawa_build
from safepath.ottawa.volumes import run_ottawa_volumes
from safepath.risk import run_score
from safepath.routing import run_routing
from safepath.snap_report import run_snap_report
from safepath.snapping import run_snap
from safepath.tracking import DISABLE_ENV

log = logging.getLogger("safepath")


def run_gnn(cfg: dict) -> None:
    """GNN training in its own process: torch and LightGBM ship different OpenMP runtimes
    on macOS and cannot both run threads in one process. Device and W&B settings pass
    through the environment."""
    cmd = [sys.executable, "-m", "safepath.gnn.train"]
    if cfg.get("_config_path"):
        cmd += ["--config", str(cfg["_config_path"])]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"GNN training exited with code {result.returncode}")


def run_app(cfg: dict) -> None:
    """Launch the Streamlit app; blocks until the server stops."""
    script = Path(__file__).parent / "app" / "streamlit_app.py"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(script)], check=False
        )
    except KeyboardInterrupt:  # Ctrl+C stops the server; that is a normal exit
        return
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
    "spf": ("Fit the negative binomial safety performance function", run_spf),
    "eb": ("Empirical Bayes expected crashes per intersection-year", run_eb),
    "evaluate": ("Evaluate models on val 2024 and test 2025", run_evaluate),
    "calibration": (
        "Check calibration by volume and recalibrate the routing model",
        run_calibration,
    ),
    "score": ("Score 2026 risk and write it onto the routing graph", run_score),
    "route": ("Calibrate k and report the routing trade-off", run_routing),
    "volumes": ("Download Toronto traffic counts, expand to AADT, match to the graph", run_volumes),
    "mto": ("Map MTO 400-series AADT and collision rates onto the graph", run_mto),
    "volume-model": ("Predict AADT for every intersection and edge", run_volume_model),
    "explain": ("SHAP for LightGBM + AADT and per-intersection risk factors", run_explain),
    "exposure": ("Write the exposure sanity report (crash rates per vehicle)", run_exposure),
    "gnn": ("Train and evaluate GNNs on the intersection graph (not in `all`)", run_gnn),
    "app": ("Launch the Streamlit routing app", run_app),
    "ottawa-data": (
        "Ottawa: download and clean collisions and intersection counts",
        lambda cfg: run_ottawa_data(ottawa_config(cfg)),
    ),
    "ottawa-build": (
        "Ottawa: graph, location check, snapping, labels and features (Toronto code paths)",
        lambda cfg: run_ottawa_build(ottawa_config(cfg)),
    ),
    "ottawa-volumes": (
        "Ottawa: Toronto's AADT model vs Ottawa counts; Ottawa AADT for the crash models",
        lambda cfg: run_ottawa_volumes(ottawa_config(cfg), toronto_config(cfg)),
    ),
}
OTTAWA_PIPELINE = ["ottawa-data", "ottawa-build", "ottawa-volumes"]
PIPELINE = [
    "download",
    "clean",
    "report-data",
    "graph",
    "locations",
    "snap",
    "report-snap",
    "features",
    "volumes",
    "mto",
    "volume-model",
    "train",
    "spf",
    "eb",
    "evaluate",
    "calibration",
    "score",
    "explain",
    "exposure",
    "route",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="safepath", description=__doc__)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--no-wandb", action="store_true", help="do not log to Weights & Biases")
    parser.add_argument("--device", default=None, help="GNN device: cuda, mps or cpu")
    sub = parser.add_subparsers(dest="step", required=True, metavar="<step>")
    for name, (help_text, _) in STEPS.items():
        sub.add_parser(name, help=help_text)
    sub.add_parser("all", help="Run the full pipeline: " + " -> ".join(PIPELINE))
    sub.add_parser("ottawa", help="Run the Ottawa cross-city test: " + " -> ".join(OTTAWA_PIPELINE))
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.no_wandb:
        os.environ[DISABLE_ENV] = "1"
    if args.device:
        os.environ["SAFEPATH_DEVICE"] = args.device
    if args.config:
        cfg["_config_path"] = args.config
    steps = {"all": PIPELINE, "ottawa": OTTAWA_PIPELINE}.get(args.step, [args.step])
    for step in steps:
        log.info("running step '%s'", step)
        try:
            STEPS[step][1](cfg)
        except Exception:
            # Stop the pipeline at the first failure; later steps must not run on stale inputs.
            log.exception("step '%s' failed; stopping", step)
            return 1
    return 0
