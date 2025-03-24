"""
BMI203: Biocomputing Algorithms - Winter 2023
Final project: neural networks
"""
from .nn import nn
from setuptools import setup, find_packages

setup(
    name="nn",
    version=("0.1.0"),
    packages=find_packages(),
    install_requires=[
        "numpy"
    ],
    author="Sophie-Christine Porak",
    description="a simple neural network built from scratch in numpy"
)
