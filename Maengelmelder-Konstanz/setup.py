from setuptools import setup, find_packages

setup(
    name="maengelmelder-konstanz",
    version="1.0.0",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.8",
)
