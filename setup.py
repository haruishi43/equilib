#!/usr/bin/env python3

import distutils.spawn
import os.path as osp
import shlex
import subprocess
import sys

from setuptools import find_packages, setup


def readme():
    with open("README.md") as f:
        content = f.read()
    return content


def find_version():
    version_file = "equilib/__init__.py"
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


def get_requirements(filename="requirements.txt"):
    here = osp.dirname(osp.realpath(__file__))
    with open(osp.join(here, filename), "r") as f:
        requires = [line.replace("\n", "") for line in f.readlines()]
    return requires


def get_long_description():
    with open("README.md") as f:
        long_description = f.read()

    try:
        import github2pypi

        return github2pypi.replace_url(
            slug="haruishi43/equilib", content=long_description
        )
    except Exception:
        return long_description


if sys.argv[1] == "release":
    if not distutils.spawn.find_executable("twine"):
        print("Please install twine:\n\n\tpip install twine\n", file=sys.stderr)
        sys.exit(1)

    commands = [
        "git pull origin master",
        "git tag v{:s}".format(find_version()),
        "git push origin master --tag",
        "python setup.py sdist",
        "twine upload dist/pyequilib-{:s}.tar.gz".format(find_version()),
    ]
    for cmd in commands:
        subprocess.check_call(shlex.split(cmd))
    sys.exit(0)


setup(
    name="pyequilib",
    version=find_version(),
    packages=find_packages(exclude=["github2pypi"]),
    description="equirectangular image processing with python",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Haruya Ishikawa",
    author_email="www.haru.ishi43@gmail.com",
    license="MIT",
    url="https://github.com/haruishi43/equilib",
    install_requires=get_requirements(),
    keywords=["Equirectangular", "Computer Vision"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.5",
)
