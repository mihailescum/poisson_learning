#!/usr/bin/env python

from setuptools import setup, find_packages

install_requires = ["numpy >= 1.21.1", "scipy >= 1.7.2", "matplotlib >= 3.4.3", "graphlearning"]

setup(
    name="poissonlearning",
    version="0.0.1",
    author="Max Mihailescu",
    packages=find_packages(),
    install_requires=install_requires,
)
