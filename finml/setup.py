from setuptools import setup, find_packages

setup(
    name="finml",
    version="0.1.1",
    description="finml -- tools for ml trading.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author='yfclark <1825617022.yf@gmail.com>',
    packages=[
        package for package in find_packages() if package.startswith('finml')
    ],
    license="Apache License 2.0",
    zip_safe=True,
    python_requires=">=3.9",
    install_requires=[
        "polars",
        "pyarrow",
        "numpy",
        "pandas",
        "scipy",
        "xgboost",
        "xgboost_ray",
        "optuna",
        "scikit-learn"
    ],
)
