"""Optional Weights & Biases experiment tracking.

Tracking is on only when the `wandb` package is installed, WANDB_API_KEY is set (in the
environment or the gitignored .env file at the repo root) and it has not been switched off
with `python -m safepath --no-wandb ...` or `tracking.enabled: false` in the config.
Otherwise every call here is a no-op, so the pipeline runs the same without it.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from safepath.config import REPO_ROOT

log = logging.getLogger(__name__)

DISABLE_ENV = "SAFEPATH_NO_WANDB"


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into the environment (existing values win)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def enabled(cfg: dict[str, Any]) -> bool:
    if os.environ.get(DISABLE_ENV) == "1" or not cfg.get("tracking", {}).get("enabled", True):
        return False
    load_dotenv()
    if not os.environ.get("WANDB_API_KEY"):
        return False
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    return True


class Run:
    """Thin wrapper; every method is a no-op when tracking is off."""

    def __init__(self, run: Any | None):
        self._run = run

    @property
    def active(self) -> bool:
        return self._run is not None

    @property
    def url(self) -> str | None:
        return self._run.url if self._run is not None else None

    def log(self, metrics: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.log(metrics)

    def summary(self, values: dict[str, Any]) -> None:
        if self._run is not None:
            for k, v in values.items():
                self._run.summary[k] = v

    def table(self, name: str, df: pd.DataFrame) -> None:
        if self._run is not None:
            import wandb

            self._run.log({name: wandb.Table(dataframe=df.reset_index(drop=True).astype(
                {c: str for c in df.columns if df[c].dtype == object}))})  # fmt: skip

    def image(self, name: str, path: Path) -> None:
        if self._run is not None and Path(path).exists():
            import wandb

            self._run.log({name: wandb.Image(str(path))})


@contextmanager
def run(
    cfg: dict[str, Any], name: str, job_type: str, config: dict[str, Any] | None = None
) -> Iterator[Run]:
    """A W&B run for one pipeline step, tagged with the git commit."""
    if not enabled(cfg):
        yield Run(None)
        return
    import wandb

    t = cfg.get("tracking", {})
    wb = wandb.init(
        project=t.get("project", "safepath"),
        entity=t.get("entity") or None,
        name=name,
        job_type=job_type,
        config={**(config or {}), "git_commit": git_commit()},
        reinit="finish_previous",
    )
    try:
        yield Run(wb)
    finally:
        wb.finish()
        log.info("logged W&B run %s: %s", name, wb.url)
