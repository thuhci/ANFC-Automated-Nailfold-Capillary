from setuptools import find_packages, setup

setup(
    name="nailfold_video_process",
    version="0.1.0",
    description="nailfold video stablization and flow estimation",
    packages=find_packages(),
    install_requires=[
        "torch"
    ]
)

