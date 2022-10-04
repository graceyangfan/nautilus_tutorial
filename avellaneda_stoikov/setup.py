#!/usr/bin/env python
import sys

from setuptools import setup

from setuptools_rust import RustExtension

setup(
    rust_extensions=[RustExtension("avellaneda_stoikov.avellaneda_stoikov")],
)