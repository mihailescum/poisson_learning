#!/usr/bin/env python

from distutils.core import setup

install_requires = [
    "numpy >= 1.21.1",
    "scipy >= 1.7.2",
    "matplotlib >= 3.4.3",
]

setup(
    name="poissonlearning",
    version="0.0.1",
    author="Max Mihailescu",
    packages=["poissonlearning"],
    install_requires=install_requires,
)

