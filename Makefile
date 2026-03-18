.PHONY: install test lint smoke demo figures clean train-quick

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/ scripts/

smoke:
	python scripts/smoke_test.py

demo:
	python scripts/quick_demo.py

figures:
	python scripts/generate_demo_figures.py --outdir paper/figures

train-quick:
	python scripts/run_paper_experiments.py --phase all

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
