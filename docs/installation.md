# Installation

## Prerequisites

- Python `>=3.9`
- PyTorch `>=2.8`

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
    `equilib` is tested against PyTorch `>=2.8`. Older releases may work but are
    not supported; if you hit a numerical or API issue, please
    [open an issue](https://github.com/haruishi43/equilib/issues).
