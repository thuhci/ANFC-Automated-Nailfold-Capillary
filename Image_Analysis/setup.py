from setuptools import find_packages, setup

setup(
    name="nailfold_image_profile",
    version="0.1.0",
    description="nailfold image process analysis",
    packages=find_packages(),
    install_requires=[
        "torch"
    ]
)

