#!/usr/bin/env python
import sys

from setuptools import setup, find_packages
from setuptools_rust import RustExtension

setup(
    name="avellaneda-stoikov",
    version="0.1.0",
    description="Avellaneda–Stoikov A/k calibrator (pyo3)",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    rust_extensions=[RustExtension("avellaneda_stoikov.avellaneda_stoikov")],
    zip_safe=False,
)
