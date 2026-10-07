"""Screenshot every dashboard tab with Playwright (and fail loudly on any app exception).

Usage::

    aerosync serve --headless --port 8599 &      # or: make serve
    python scripts/capture_dashboard.py --url http://localhost:8599
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

TABS = ["Live Simulation", "Gate Timeline", "Disruption Lab", "Explainability", "Benchmark",
        "About / PEAS"]
OUT = Path("docs/slide_assets/screenshots")


def wait_idle(page: Page, timeout: float = 90.0) -> None:
    """Wait until Streamlit stops showing its running indicator."""
    t0 = time.time()
    time.sleep(1.0)
    while time.time() - t0 < timeout:
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            time.sleep(0.8)
            if page.locator('[data-testid="stStatusWidget"]').count() == 0:
                return
        time.sleep(0.5)


def check_errors(page: Page, where: str) -> None:
    errs = page.locator('[data-testid="stException"]')
    if errs.count():
        print(f"ERROR in {where}:\n{errs.first.inner_text()}", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8599")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1680, "height": 1050}, device_scale_factor=1)
        page.goto(args.url, wait_until="networkidle")
        page.get_by_role("tab").first.wait_for(timeout=120_000)
        wait_idle(page)
        for i, name in enumerate(TABS, start=1):
            page.get_by_role("tab", name=name).click()
            wait_idle(page)
            if name == "Live Simulation":
                page.get_by_role("button", name="▶ Play").click()
                time.sleep(6)
                page.get_by_role("button", name="⏸ Pause").click()
                wait_idle(page)
            if name == "Disruption Lab":
                page.get_by_role("button", name="Inject fault and run").click()
                wait_idle(page, 180)
            check_errors(page, name)
            slug = name.lower().replace(" / ", "_").replace(" ", "_")
            path = out / f"dashboard_{i}_{slug}.png"
            page.screenshot(path=str(path), full_page=True)
            print(f"saved {path}")
        browser.close()


if __name__ == "__main__":
    main()
