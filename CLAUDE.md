# Equilib

`equilib` (PyPI: `pyequilib`) is a Python library for processing equirectangular
(360°) images. It provides transforms between equirectangular, cubemap, and
perspective representations (`cube2equi`, `equi2cube`, `equi2equi`, `equi2pers`,
`pers2equi`), each with both a `class` and a `func` API. Inputs may be
`numpy.ndarray` or `torch.Tensor` (channel-first, `BxCxHxW` or `CxHxW`) and the
output type matches the input.

Layout:
- `equilib/<transform>/` — each transform has `base.py` (dispatch + public API),
  `numpy.py`, and `torch.py` backends.
- `equilib/grid_sample/` — numpy/torch (and WIP cpp) grid-sampling backends.
- `equilib/{numpy,torch}_utils/` — rotation/intrinsic/grid helpers.
- `tests/` — pytest suite (`test_*`) plus shared `helpers/` and the
  `grid_sample/` reference package imported by the grid-sample tests.
- `benchmarks/` — standalone (non-pytest) performance/baseline scripts.
- `docs/` + `mkdocs.yml` — MkDocs Material documentation.

Tooling: [uv](https://docs.astral.sh/uv/) for env/deps, [Ruff](https://docs.astral.sh/ruff/)
for lint/format, pytest for tests, Git LFS for image/video assets. Common
commands: `uv sync --group dev`, `uv run pytest tests`, `uv run ruff check .`,
`uv run --group docs mkdocs serve`.

## Behavioral Guidelines for Coding Tasks

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
