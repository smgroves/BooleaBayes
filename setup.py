"""
package setup
@author: SM Groves
"""
import sys
import os
import io
import setuptools
from setuptools import setup


def read(fname):
    with io.open(
        os.path.join(os.path.dirname(__file__), fname), encoding="utf-8"
    ) as _in:
        return _in.read()


if __name__ == "__main__":

    with open("README.md", "r") as fh:
        long_description = fh.read()

    setup(
        name="booleabayes",
        version="0.0.2",
        description="A suite for network inference from transcriptomics data",
        long_description=long_description,
        author="Sarah Groves",
        author_email="sarahmaddoxgroves@gmail.com",
        url="https://github.com/smgroves/BooleaBayes",
        install_requires=[
            "seaborn",
            "anndata",
            "leidenalg",
            "matplotlib",
            "pandas",
            "umap-learn",
            "numpy",
            "scipy",
            "cython",
            "numba",
            "scikit-learn",
            "h5py",
            "click",
            "magic-impute",
            "networkx",
        ],
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering",
        ],
        python_requires=">=3.6",
    )
