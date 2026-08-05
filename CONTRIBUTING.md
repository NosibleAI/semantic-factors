# Contributing

Thank you for helping improve `semantic-factors`. The package is a client for the grouped
NOSIBLE World search mode; factor arithmetic and analysis belong in this repository, while
backend changes belong in the World service.

## Local setup

Use a supported Python version (3.10-3.14) and install the development dependencies into a
virtual environment:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the same checks used by CI before opening a pull request:

```bash
python -m pytest
python -m ruff check .
python -m mypy src
python -m build
python -m twine check dist/*
semantic-factors --version
semantic-factors fields --search materiality
semantic-factors schemes
```

Tests must remain offline and must not contain customer data, World responses, API keys, cache
files, generated factor outputs, or proprietary NOSIBLE World data. Keep public examples
reproducible without credentials or charged network requests.

## Pull requests

Keep changes focused, explain user-visible behavior in the pull request description, and add or
update regression tests for behavior changes. Do not change the grouped World API contract from
this repository without coordinating with the backend owner. Release and publication changes
require maintainer approval.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). For security
reports, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.
