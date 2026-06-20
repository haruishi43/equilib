# Installation

## Prerequisites

- Python `>=3.9`
- PyTorch (a recent release; see the note below)

## Install from PyPI

```bash
pip install pyequilib
```

## Development install

This project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management.

```bash
git clone https://github.com/haruishi43/equilib.git
cd equilib

# create the virtual environment and install the package + dev tools
uv sync --group dev
```

Run common tasks through uv:

```bash
uv run pytest tests          # run the test suite
uv run ruff check .          # lint
uv run ruff format .         # format
```

!!! note
    `equilib` was originally developed and tested against PyTorch 1.12. Newer
    PyTorch releases generally work, but if you hit a numerical or API issue,
    please [open an issue](https://github.com/haruishi43/equilib/issues).
