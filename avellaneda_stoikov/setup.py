from setuptools import setup

from setuptools_rust import RustBin

setup(
    name="avellaneda_stoikov",
    version="1.0",
    rust_extensions=[
        RustBin(
            "avellaneda_stoikov",
            args=["--profile", "release-lto"],
        )
    ],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)