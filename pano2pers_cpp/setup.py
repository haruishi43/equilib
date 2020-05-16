#/usr/bin/env python3

from setuptools import setup, Extension
from torch.utils import cpp_extension


setup(
    name='pano2pers_cpp',
    ext_modules=[
        cpp_extension.CppExtension(
            'demo_cpp',
            ['demo.cc']
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension})
