# Contributing to equilib

Thanks for your interest in improving `equilib`! Bug reports, fixes, features,
and documentation improvements are all welcome.

## Issues

Use [GitHub issues](https://github.com/haruishi43/equilib/issues) for bugs and
feature requests. For bugs, please include clear steps to reproduce.

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) for environments and
dependencies, [Ruff](https://docs.astral.sh/ruff/) for linting and formatting,
and [Git LFS](https://git-lfs.com/) for image/video assets.

```bash
git lfs install             # once per machine, before cloning
git clone https://github.com/haruishi43/equilib.git
cd equilib

uv sync --group dev         # create the venv and install package + dev tools
uv run pre-commit install   # optional: run Ruff automatically on each commit
```

## Tests

Make sure the test suite passes before submitting:

```bash
uv run pytest tests
```

## Coding style

- We follow PEP8 and use [typing](https://docs.python.org/3/library/typing.html).
- Linting and formatting are handled by Ruff (and run automatically by the
  pre-commit hooks):

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
```

## Documentation

Documentation is built with
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/). Preview it
locally:

```bash
uv run --group docs mkdocs serve
```

## Pull requests

1. Branch off `master`.
2. Keep changes focused; update tests and docs alongside code.
3. Ensure `uv run ruff check .`, `uv run ruff format --check .`, and
   `uv run pytest tests` all pass.
4. Open the PR with a clear description of the change.

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

## Roadmap

- [ ] Better type hints
- [ ] Type hints for `tests`
