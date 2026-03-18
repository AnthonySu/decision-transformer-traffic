# Contributing to EV-DT

Thank you for your interest in contributing. This document covers the development setup, code standards, and process for submitting changes.

## Development Setup

```bash
# Clone and install with dev dependencies
git clone https://github.com/yourusername/decision-transformer-traffic.git
cd decision-transformer-traffic
pip install -e ".[dev]"
```

The `[dev]` extra installs: `pytest`, `ruff`, `black`, and `ipython`.

## Code Style

This project uses **ruff** for linting and **black** for formatting, both configured in `pyproject.toml`.

```bash
# Format code
black src/ scripts/ tests/

# Lint
ruff check src/ scripts/ tests/

# Lint and auto-fix
ruff check --fix src/ scripts/ tests/
```

Configuration summary (from `pyproject.toml`):
- **Line length:** 100 characters
- **Target version:** Python 3.10
- **Ruff rules:** E (pycodestyle errors), F (pyflakes), I (isort), W (pycodestyle warnings)

Please run both tools before submitting a pull request. CI will reject PRs that fail lint or format checks.

## Testing

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_network_utils.py -v
```

When adding new functionality:
- Add unit tests in the `tests/` directory.
- Test files should be named `test_<module>.py`.
- Ensure existing tests still pass before submitting.

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`.
2. **Make your changes.** Keep commits focused and atomic.
3. **Run checks locally:**
   ```bash
   black --check src/ scripts/ tests/
   ruff check src/ scripts/ tests/
   pytest tests/
   ```
4. **Write a clear PR description** explaining what the change does and why.
5. **Submit the PR** against `main`. A maintainer will review it.

## Adding a New Baseline

To add a new baseline method:

1. Create a policy class in `src/baselines/` (see `greedy_preempt.py` for the expected interface).
2. Register it in `scripts/evaluate.py` so it is included in the evaluation sweep.
3. Add any new config parameters under the `baselines` section of `configs/default.yaml`.
4. Add a training script entry in `scripts/` if the baseline requires training.

## Adding a New Metric

1. Implement a `compute_<metric_name>()` function in `src/utils/metrics.py`.
2. Wire it into `aggregate_metrics()` in the same file.
3. Add the metric name to the `eval.metrics` list in `configs/default.yaml`.

## Questions

Open an issue on GitHub for bug reports, feature requests, or questions about the codebase.
