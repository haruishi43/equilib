# Contributing to equilib

I want to make contributing to this project as easy and transparent as possible.

## Issues

See [GitHub issues](../../issues) to track public bugs. Please ensure your description is clear and has sufficient instructions to be able to reproduce the issue.

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) for environments and
dependencies, and [Ruff](https://docs.astral.sh/ruff/) for linting and
formatting.

```bash
uv sync --group dev
uv run pre-commit install   # optional: run Ruff automatically on commit
```

## Test

Use the pytest testing framework and make sure the tests pass before submitting:

```bash
uv run pytest tests
```

## Coding Style

- We follow PEP8 and use [typing](https://docs.python.org/3/library/typing.html).
- Linting and formatting are handled by Ruff:

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
```

Pre-commit hooks run the same Ruff checks. Install them with
`uv run pre-commit install`.

## Documentation

Documentation is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).
Preview it locally:

```bash
uv run --group docs mkdocs serve
```

## Large files

Image and video assets are stored with [Git LFS](https://git-lfs.com/). Install
it once (`git lfs install`) before cloning or committing binary assets.

## TODO

- [ ] Better type hints
- [ ] Type hints for `tests`
