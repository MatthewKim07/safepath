"""torch and LightGBM each load their own libomp on macOS (torch's bundled copy, LightGBM's
Homebrew one); a process that runs threads in both segfaults inside libomp. The pipeline
keeps them apart: no LightGBM step imports torch, and GNN training runs in a subprocess.
These checks run each side in a fresh process."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True,
                          text=True, timeout=600)  # fmt: skip


def test_pipeline_process_never_imports_torch_and_lightgbm_runs():
    r = run(
        "import sys\n"
        "import safepath.cli, safepath.models.train, safepath.evaluate, safepath.risk\n"
        "import safepath.models.calibration, safepath.explain, safepath.routing\n"
        "import lightgbm as lgb, numpy as np\n"
        "X = np.random.default_rng(0).random((20000, 10))\n"
        "y = np.random.default_rng(1).poisson(np.exp(X[:, 0]))\n"
        "lgb.train({'objective': 'poisson', 'verbosity': -1}, lgb.Dataset(X, y), 30)\n"
        "assert 'torch' not in sys.modules, 'a LightGBM pipeline process imported torch'\n"
        "print('ok')\n"
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == "ok"


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="gnn extra not installed")
def test_gnn_tests_pass_in_their_own_process():
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:warnings", "-p", "no:cacheprovider",
         "-o", "addopts=", "tests/gnn"],
        cwd=REPO, capture_output=True, text=True, timeout=1200,
    )  # fmt: skip
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]


def test_gnn_cli_step_uses_a_subprocess():
    from safepath import cli

    src = Path(cli.__file__).read_text()
    assert '"-m", "safepath.gnn.train"' in src
    assert "import safepath.gnn" not in src and "from safepath.gnn" not in src
