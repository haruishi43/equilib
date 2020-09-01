#!/usr/bin/env python3

import os.path as osp

from setuptools import find_packages, setup


def readme():
    with open('README.md') as f:
        content = f.read()
    return content


def find_version():
    version_file = 'equilib/__init__.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def get_requirements(filename='requirements.txt'):
    here = osp.dirname(osp.realpath(__file__))
    with open(osp.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires


setup(
    name='equilib',
    version=find_version(),
    description='equirectangular image processing with python',
    author='Haruya Ishikawa',
    author_email="www.haru.ishi43@gmail.com",
    license='MIT',
    long_description=readme(),
    url='https://github.com/haruishi43/equilib',
    packages=find_packages(),
    install_requires=get_requirements(),
    keywords=['Equirectangular', 'Computer Vision'],
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
    python_requires='>=3.5',
)
