"""
package setup
@author: SM Groves
"""
import sys
import os
import io
import setuptools
from setuptools import setup

install_requires = [
    "requests",
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
]

doc_requires = [
    "sphinx",
    "sphinxcontrib-napoleon",
]


def read(fname):
    with io.open(
        os.path.join(os.path.dirname(__file__), fname), encoding="utf-8"
    ) as _in:
        return _in.read()


readme = open("README.rst").read()


setup(
    name="booleabayes",
    version="0.1.9",
    description="A suite for network inference from transcriptomics data",
    long_description=readme,
    long_description_content_type="text/x-rst",
    author="Sarah Groves",
    author_email="sarahmaddoxgroves@gmail.com",
    url="https://github.com/smgroves/BooleaBayes",
    install_requires=install_requires,
    extras_require={"doc": doc_requires},
    packages=setuptools.find_packages(exclude = "tests"),
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
