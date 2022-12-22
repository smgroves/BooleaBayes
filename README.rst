=======================================================
BooleaBayes
=======================================================
.. image:: https://badge.fury.io/py/booleabayes.svg
    :target: https://pypi.org/project/booleabayes/
    :alt: Latest PYPi Version

BooleaBayes is a suite of network inference tools to derive and simulate gene regulatory networks from transcriptomics data. To see how BooleaBayes could be applied to your own data, see our publication in PLOS Computational Biology, `Wooten, Groves et al. (2019) <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007343>`_. 


Installation
~~~~~~~~~~~~~~~~~

To install ``BooleaBayes``, please use::

    pip install booleabayes

Dependencies
---------------------

The ``graph-tool`` python package will need to be installed. This can be installed with `Conda`, `homebrew`, etc as seen `here <https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions>`_. All other dependencies will be installed with this package.

BooleaBayes Usage:
~~~~~~~~~~~~~~~~~~~~~~~~

* ``net`` = make or modify network structure
* ``load`` = loading data
* ``proc`` = processing
* ``rw`` = random walk
* ``plot`` = plotting
* ``tl`` = tools
* ``utils`` = utilities

BooleaBayes Tutorial:
~~~~~~~~~~~~~~~~~~~~~~~~

A tutorial can be found `here <https://github.com/smgroves/BooleaBayes/blob/main/Tutorials/bbayes_sample.ipynb>`_. This tutorial assumes you have already generated a network structure and want to use BooleaBayes to fit the network rules and run simulations.
The network structure should be a csv as outputted by ``bb.net.make_network()``, where the first column is the parent nodes and the second column is the child nodes. Additional columns of metadata can also be given, but the file should have no headers.

