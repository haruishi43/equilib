#!/usr/bin/env python3

import os.path as osp

import numpy as np

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


def numpy_include():
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include


def get_requirements(filename='requirements.txt'):
    here = osp.dirname(osp.realpath(__file__))
    with open(osp.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires


setup(
    name='equilib',
    version=find_version(),
    description='',
    author='Haruya Ishikawa',
    license='',
    long_description=readme(),
    url='',
    packages=find_packages(),
    install_requires=get_requirements(),
    keywords=['Equirectangular', 'Computer Vision'],
)
