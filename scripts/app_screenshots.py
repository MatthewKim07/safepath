"""Launch the Streamlit app headless, time it, measure its memory, and screenshot preset
routes in both cities.

Usage: python scripts/app_screenshots.py [--out reports/figures] [--port 8599]
       [--python /path/to/venv/bin/python] [--app deploy/streamlit_app.py]

Memory is the Streamlit server process's resident set size (ps), sampled every 0.2 s.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "deploy" / "streamlit_app.py"
TORONTO = {"city": "toronto", "origin": "Union Station", "destination": "Yorkdale"}
LONDON = {"city": "london", "origin": "Western University (UCC)",
          "destination": "White Oaks Mall"}  # fmt: skip
SHOTS = [
    ("app_toronto_lambda0.png", {**TORONTO, "lam": "0"}),
    ("app_toronto_lambda2.png", {**TORONTO, "lam": "2"}),
    ("app_london_lambda0.png", {**LONDON, "lam": "0"}),
    ("app_london_lambda2.png", {**LONDON, "lam": "2"}),
]


class PeakRSS:
    """Peak resident memory (MB) of one process, polled in the background."""

    def __init__(self, pid: int):
        self.pid, self.peak, self.now = pid, 0.0, 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            out = subprocess.run(["ps", "-o", "rss=", "-p", str(self.pid)],
                                 capture_output=True, text=True).stdout.strip()  # fmt: skip
            if out:
                self.now = int(out) / 1024
                self.peak = max(self.peak, self.now)
            time.sleep(0.2)

    def stop(self) -> None:
        self._stop.set()
        self._t.join()


def wait_healthy(port: int, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/_stcore/health") as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError("streamlit did not become healthy")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "reports" / "figures"))
    parser.add_argument("--port", type=int, default=8599)
    parser.add_argument("--python", default=sys.executable, help="interpreter to run Streamlit")
    parser.add_argument("--app", default=str(APP))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    server = subprocess.Popen(
        [
            args.python,
            "-m",
            "streamlit",
            "run",
            args.app,
            "--server.headless",
            "true",
            "--server.port",
            str(args.port),
            "--browser.gatherUsageStats",
            "false",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=ROOT,
    )
    mem = PeakRSS(server.pid)
    try:
        wait_healthy(args.port)
        print(f"server healthy after {time.time() - t0:.1f}s, {mem.now:.0f} MB")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            for i, (name, params) in enumerate(SHOTS):
                query = "&".join(f"{k}={v.replace(' ', '+')}" for k, v in params.items())
                start = time.time()
                page.goto(f"http://localhost:{args.port}/?{query}")
                page.get_by_text("Fastest route").first.wait_for(timeout=60_000)
                page.frame_locator("iframe").first.locator(".leaflet-interactive").first.wait_for(
                    timeout=60_000
                )
                page.wait_for_timeout(1500)  # let tiles finish drawing
                elapsed = time.time() - start
                label = "cold start (server + first render)" if i == 0 else "page render"
                total = time.time() - t0 if i == 0 else elapsed
                print(f"{name}: {label} {total:.1f}s; server RSS {mem.now:.0f} MB "
                      f"(peak {mem.peak:.0f} MB)")  # fmt: skip
                page.screenshot(path=str(out / name), full_page=True)
            browser.close()
        print(f"peak server memory: {mem.peak:.0f} MB")
    finally:
        mem.stop()
        server.terminate()
        server.wait(timeout=10)
    if server.returncode not in (0, -15):
        print(server.stderr.read().decode(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
