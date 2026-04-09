# pyseroepi 🦠🧬🗺️
###### A Python library for Pathogen Genotype eXploration

[![License](https://img.shields.io/badge/license-GPLv3-blue.svg?style=flat-square&maxAge=2678400)](https://choosealicense.com/licenses/gpl-3.0/)
[![PyPI](https://img.shields.io/pypi/v/pyseroepi.svg?style=flat-square&maxAge=3600&logo=PyPI)](https://pypi.org/project/pyseroepi)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/pyseroepi?style=flat-square&maxAge=3600&logo=anaconda)](https://anaconda.org/bioconda/pyseroepi)
[![Wheel](https://img.shields.io/pypi/wheel/pyseroepi.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/pyseroepi/#files)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyseroepi.svg?style=flat-square&maxAge=600&logo=python)](https://pypi.org/project/pyseroepi/#files)
[![Python Implementations](https://img.shields.io/pypi/implementation/pyseroepi.svg?style=flat-square&maxAge=600&label=impl)](https://pypi.org/project/pyseroepi/#files)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square)](https://github.com/tomdstanton/pyseroepi/)
[![Issues](https://img.shields.io/github/issues/tomdstanton/pyseroepi.svg?style=flat-square&maxAge=600)](https://github.com/tomdstanton/pyseroepi/issues)
[![Docs](https://img.shields.io/readthedocs/pyseroepi/latest?style=flat-square&maxAge=600)](https://pyseroepi.readthedocs.io)
[![Changelog](https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square)](https://github.com/tomdstanton/pyseroepi/blob/main/CHANGELOG.md)
[![Downloads](https://img.shields.io/pypi/dm/pyseroepi?style=flat-square&color=303f9f&maxAge=86400&label=downloads)](https://pepy.tech/project/pyseroepi)

> [!WARNING]
> 🚧 This package is currently under construction, proceed with caution 🚧

## Introduction 🌐
`pyseroepi` is a Python library for Pathogen Genotype eXploration. It started a Python port of the 
[KleborateR](https://github.com/klebgenomics/KleborateR) R code to parse 
[Kleborate](https://github.com/klebgenomics/Kleborate) and other data from [Pathogenwatch](https://pathogen.watch/),
to calculate prevalence data for sero-epidemiology, and to provide a backend for our 
[neonatal sepsis sero-epi app](https://github.com/klebgenomics/KlebNNSapp).

`pyseroepi` aims to be more generalised towards other bacterial genotyping and distance calculation methods, and
will eventually force complicity with the 
[PHA4GE genotyping-specification](https://github.com/pha4ge/genotyping-specification).

`pyseroepi` revolves around genotyping `Dataset` objects, which have the following attributes:

- Genotyping data - a `pandas` dataframe containing the parsed output of a genotyping tool. 
- Optional metadata - joined to genotyping results upon initialisation, possibly containing spatio-temporal data.
- Optional distances - Represented as a `scipy.sparse` matrix of pairwise distances, parsed from the outputs of tools 
such as `mash`.

We also define `Calculator`s, which calculate informative information from genotyping data, such as prevalence and 
diversity.

## Installation ⚙️

### From source:
```shell
pip install git+https://github.com/tomdstanton/pyseroepi.git
```

## Usage 🧑‍💻
The information below explains how to use the `pyseroepi` CLI. 
For API usage, please refer to the [reference documentation](https://tomdstanton.github.io/pyseroepi/reference/pyseroepi/).

### Prevalence 🌎
The `PrevalenceCalculator` object to calculate prevalence from a dataset is exposed via the command-line. For more
information about the calculator, please refer to the 
[API docs](https://tomdstanton.github.io/pyseroepi/reference/pyseroepi/calculators/#pyseroepi.calculators.PrevalenceCalculator).

