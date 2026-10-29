"""Launch the Streamlit app headless, time it, and screenshot preset routes.

Usage: python scripts/app_screenshots.py [--out reports/figures] [--port 8599]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "safepath" / "app" / "streamlit_app.py"
SHOTS = [
    ("app_lambda0.png", {"origin": "Union Station", "destination": "Yorkdale", "lam": "0"}),
    ("app_lambda2.png", {"origin": "Union Station", "destination": "Yorkdale", "lam": "2"}),
]


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
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP),
            "--server.headless",
            "true",
            "--server.port",
            str(args.port),
            "--browser.gatherUsageStats",
            "false",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        wait_healthy(args.port)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900})
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
                print(f"{name}: {label} {total:.1f}s")
                page.screenshot(path=str(out / name), full_page=True)
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)
    if server.returncode not in (0, -15):
        print(server.stderr.read().decode(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
