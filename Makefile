# AeroSync - common tasks. On Windows run these from Git Bash, or use the
# equivalent commands shown in the README.
PY ?= python
SEEDS ?= 42,43,44,45,46

.PHONY: install demo test lint report assets screenshots serve clean

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	$(PY) -m playwright install chromium

demo:
	bash demo.sh

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

report:
	$(PY) -m aerosync report --scenario all --out results/ --seeds $(SEEDS)

assets: report
	$(PY) scripts/make_slide_assets.py

screenshots:
	$(PY) scripts/capture_dashboard.py --url http://localhost:8599

serve:
	$(PY) -m aerosync serve

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info results/compare
