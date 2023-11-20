from setuptools import find_packages, setup

setup(
    name="image_segmentation",
    version="0.1.0",
    description="nailfold image segmentation",
    packages=find_packages(),
    install_requires=[
        "torch"
    ]
)

