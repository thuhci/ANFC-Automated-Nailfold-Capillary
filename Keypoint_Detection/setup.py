from setuptools import find_packages, setup

setup(
    name="nailfold_keypoint",
    version="0.1.0",
    description="nailfold keypoint detector",
    packages=find_packages(),
    install_requires=[
        "torch"
    ]
)

