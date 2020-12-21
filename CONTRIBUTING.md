# Contributing to equilib
We want to make contributing to this project as easy and transparent as possible.

## Issues
We use [GitHub issues](../../issues) to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## Test
We use pytest testing framework and testing data that needs to be downloaded, please make sure that test are passing:
```
pytest
```

## Check typing
We use mypy to check Python typing and guard API consistency, please make sure next command doesn't complain prior to submission:
```
mypy . --ignore-missing-imports
```

## Coding Style
  - We follow PEP8 and use [typing](https://docs.python.org/3/library/typing.html).
  - Use `black` for style enforcement and linting. Install black through `pip install black`.

  We also use pre-commit hooks to ensure linting and style enforcement. Install the pre-commit hooks with `pip install pre-commit && pre-commit install`.

## Documentation
WIP
