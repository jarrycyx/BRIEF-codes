.. Condensa documentation master file, created by
   sphinx-quickstart on Tue Sep  4 15:17:30 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Condensa Documentation
======================

Condensa is a framework for **programmable model compression** in Python.
It comes with a set of built-in compression operators which may be used to
compose complex compression schemes targeting specific combinations of DNN,
hardware platform, and optimization objective.
Common programming abstractions such as conditionals, iteration, and
recursion are all natively supported.
To recover any accuracy lost during compression, Condensa uses a constrained
optimization formulation of model compression and employs an Augmented Lagrangian-based
algorithm as the optimizer.

Condensa is under active development, and bug reports, pull requests, and other feedback are all highly appreciated.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   guide/install
   guide/usage

.. toctree::
   :maxdepth: 2
   :caption: Module API Reference

   modules/schemes
   modules/pi
   modules/compressor
   modules/opt
   modules/finetuner
   modules/tensor
   modules/functional
   modules/util

.. toctree::
   :maxdepth: 2
   :caption: Notes

   modules/lc

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
