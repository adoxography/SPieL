# [SPieL](https://github.com/adoxography/SPieL)

[![Build Status](https://scrutinizer-ci.com/g/adoxography/SPieL/badges/build.png?b=master)](https://scrutinizer-ci.com/g/adoxography/SPieL/build-status/master)
[![Code Coverage](https://scrutinizer-ci.com/g/adoxography/SPieL/badges/coverage.png?b=master)](https://scrutinizer-ci.com/g/adoxography/SPieL/?branch=master)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/adoxography/SPieL/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/adoxography/SPieL/?branch=master)

SPieL stands for "Segmentation of Polysynthetic Languages." It is a tool for splitting words from highly inflected [polysynthetic](https://en.wikipedia.org/wiki/Polysynthetic_language) languages into their constituent [morphemes](https://en.wikipedia.org/wiki/Morpheme). It is still very much in pre-alpha stage.

## Installation
SPieL requires Python &ge; 3.6. Assuming you have it set up:
1. Clone the repository

```bash
git clone https://github.com/adoxography/SPieL
```

2. Install the package

```bash
python setup.py install --user
```

If you need to uninstall it later, `pip uninstall spiel` will remove it from your system.

## Usage

### Command line
SPieL sets up a command line utility, `spiel`, when it is installed. It can be invoked like so:

```bash
spiel --train TRAIN_FILE [--test TEST_FILE]
```

`TRAIN_FILE` and `TEST_FILE` must correspond to text files with instance data prepared SPieL's expected format. (See below.)

### Instance file format
Instances may be given either in sets of three lines, or in single lines. Three line instances should be structured as follows:

```
Shape
segment ation of   shape
label   for   each segment
```

The final two lines may be omitted, but files with instances structured in such a way may only be used as the `--test` argument.

## References
* Antal van den Bosch and Sander Canisius. 2006. [Improved morpho-phonological sequence processing with constraint satisfaction inference](http://aclweb.org/anthology/W06-3206). In *Proceedings of the Eighth Meeting of the ACL Special Interest Group on Computational Phonology and Morphology at HLT-NAACL 2006*, pages 41-49. Association for Computational Linguistics.
