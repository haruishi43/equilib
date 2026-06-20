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

## Releasing (maintainers)

Releases are published to PyPI automatically by the
[`python-publish.yml`](.github/workflows/python-publish.yml) workflow when a
GitHub Release is **published**. Publishing uses
[trusted publishing (OIDC)](https://docs.pypi.org/trusted-publishers/) — there is
no PyPI API token.

### One-time setup

Configure a trusted publisher for the `pyequilib` project at
<https://pypi.org/manage/project/pyequilib/settings/publishing/> with:

- Owner / repository: `haruishi43/equilib`
- Workflow filename: `python-publish.yml`
- Environment: `pypi`

The publish job runs in a `pypi` GitHub Environment; create it under the repo's
*Settings → Environments* if you want to attach approval/protection rules
(optional, but the environment name must match the value above).

### Per release

1. Bump `__version__` in [`equilib/info.py`](equilib/info.py) — the single
   source of truth that `pyproject.toml` reads. Use a
   [PEP 440](https://peps.python.org/pep-0440/) version; e.g. `0.6.0rc1` for a
   release candidate, `0.6.0` for the final release.
2. Refresh the lockfile and citation: `uv lock`, and update the `version` in the
   README/docs citation for a final release.
3. Verify locally:
   ```bash
   uv run ruff check . && uv run ruff format --check .
   uv run pytest tests
   uv build && uv run --with twine twine check dist/*
   ```
4. Open a PR with the bump, get it merged to `master`.
5. Create a Git tag matching the version and a GitHub Release from it
   (e.g. tag `v0.6.0rc1`). **Mark it as a pre-release for `rc`/`a`/`b`
   versions.** Publishing the release triggers the workflow, which builds the
   wheel + sdist and uploads them to PyPI.

Pre-releases are not installed by `pip install pyequilib`; users must opt in with
`pip install --pre pyequilib` (or pin the exact version).

## TODO

- [ ] Better type hints
- [ ] Type hints for `tests`
