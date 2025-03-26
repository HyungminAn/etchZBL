from setuptools import setup, find_packages

#try:
#    import torch
#except ImportError:
#    print("PyTorch is not installed. Please install it first.")

setup(
    name="PlasmaEtchSimulator",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "ase==3.23.0",
        "numpy==1.26.4",
        "networkx==3.2.1",
        "PyYAML==6.0.2",
        "torch==2.2.2",
        ]
)
