from setuptools import find_packages, setup

setup(
    name="nailfold_classifier",
    version="0.1.0",
    description="nailfold image classifier",
    packages=find_packages(),
    install_requires=[
        "torch"
    ]
)

