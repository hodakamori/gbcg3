"""Minimal setup file for tasks project."""

from setuptools import setup, find_packages

setup(
    name="gbcg3",
    version="0.1.0",
    license="MIT",
    install_requires=["numpy"],
    description="graph based coarse graining",
    author="hodaka mori",
    author_email="kpnhodaka@gmail.com",
    packages=find_packages(where="gbcg3"),
    package_dir={"": "gbcg3"},
)
