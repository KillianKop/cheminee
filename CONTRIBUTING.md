# Contributing to cheminee

Thanks for helping! This project is small and friendly: open an issue to discuss an idea, or a draft pull
request early if you prefer to show code.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: avoids the large CUDA build
pip install -e ".[dev]"
tar -xzf data/dem/sample/rasters_COP30_sample.tar.gz -C data/dem/sample/
pytest
```

## Before opening a pull request

```bash
ruff check . && ruff format --check .
pytest
```

CI runs the same checks on Python 3.10 and 3.13, plus a smoke run of `cheminee plan` and `cheminee export` on the
sample DEM. A PR is merged when CI is green and one other contributor has approved it.

- One topic per PR. Small PRs get reviewed faster.
- Branch names: `feature/...`, `fix/...`, `docs/...`.
- Commit messages: imperative mood, short first line ("Add snow line to the GUI sliders"), details in the body.
- Add or update tests with the change. Bug fixes come with the test that reproduces the bug.
- Update the docs in the same PR: README tables for new options, [docs/model.md](docs/model.md) for formulas,
  [docs/architecture.md](docs/architecture.md) for new modules.
- Add a line under "Unreleased" in [CHANGELOG.md](CHANGELOG.md) for anything a user would notice.

## Code style

- English everywhere: names, docstrings, comments, logs, messages, docs.
- Type hints on public functions; vectorised numpy on grids.
- Respect the dependency rule described in [docs/architecture.md](docs/architecture.md): the domain core imports
  nothing from the outer layers.
- Logs are `print()` lines with a bracketed prefix (`[search] 5 routes kept`).

## Changing the flight package

The flight package is the contract with the **vautour** app. Read "Changing the format" in
[docs/flight-package.md](docs/flight-package.md) first. In short: update the schema, the doc, the tests and the
changelog together, then open the matching PR in vautour with the copied schema and the regenerated sample.

## Data

- Never commit IGC tracks of real pilots: they are personal data. Share them privately, with the pilots'
  consent, if they are needed for training.
- New sample data must come with its licence and attribution in `data/README.md`.

## Licence

By contributing you agree that your contributions are released under the [MIT licence](LICENSE).
