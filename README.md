[![DOI](https://zenodo.org/badge/192713860.svg)](https://zenodo.org/badge/latestdoi/192713860)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04880/status.svg)](https://doi.org/10.21105/joss.04880)
[![PyPi](https://badgen.net/pypi/v/abcvoting)](https://pypi.org/project/abcvoting/)
![Python versions](https://badgen.net/pypi/python/abcvoting)
[![Build badge](https://github.com/martinlackner/abcvoting/workflows/Build/badge.svg)](https://github.com/martinlackner/abcvoting/actions)
[![Unittests badge](https://github.com/martinlackner/abcvoting/workflows/Unittests/badge.svg)](https://github.com/martinlackner/abcvoting/actions)
[![docs](https://readthedocs.org/projects/abcvoting/badge/?version=latest&style=plastic)](https://abcvoting.readthedocs.io/en/latest/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/martinlackner/abcvoting/branch/master/graph/badge.svg)](https://codecov.io/gh/martinlackner/abcvoting)

# abcvoting

> [!NOTE]
> - [**Documentation**](https://abcvoting.readthedocs.io/)
> - [**Installation**](https://abcvoting.readthedocs.io/en/latest/installation.html)
> - [**How to cite the abcvoting library**](https://abcvoting.readthedocs.io/en/latest/howtocite.html)
> - [**Acknowledgements and contributors**](https://abcvoting.readthedocs.io/en/latest/acks.html)
> - [**Benchmark Results**](https://martinlackner.github.io/abcvoting/benchmark/)
> - [**► abcvoting web app**](https://pref.tools/abcvoting/)

For an overview of other software tools related to Computational Social Choice, see the [COMSOC community page](https://comsoc-community.org/tools).

## A Python library of approval-based committee (ABC) rules

**A**pproval-**b**ased **c**ommittee rules (ABC rules) are voting methods for selecting a committee, i.e., a fixed-size subset of candidates.
ABC rules are also known as approval-based multi-winner rules.
The input of such rules are
[approval ballots](https://en.wikipedia.org/wiki/Approval_voting#/media/File:Approval_ballot.svg).
We recommend the book
([Multi-Winner Voting with Approval Preferences](https://link.springer.com/book/10.1007/978-3-031-09016-5))
by Lackner and Skowron [2] as a detailed introduction to ABC rules and related research directions.
In addition, the
[survey by Faliszewski et al.](http://research.illc.uva.nl/COST-IC1205/BookDocs/Chapters/TrendsCOMSOC-02.pdf) [1]
is useful as a more general introduction to committee voting (not limited to approval ballots).

The following ABC rules are implemented:

* Approval Voting (AV)

* Satisfaction Approval Voting (SAV)

* Proportional Approval Voting (PAV)

* Sequential Proportional Approval Voting (seq-PAV)

* Reverse Sequential Proportional Approval Voting (revseq-PAV)

* Approval Chamberlin-Courant (CC)

* Phragmén's sequential rule

* Monroe's rule

* Minimax Approval Voting (MAV)

* Greedy Monroe

* Method of Equal Shares (Rule X)

* Phragmén's First Method (Enestr&ouml;m's Method)

* and many more ...

In addition, one can verify axiomatic properties such as

* Justified Representation (JR)

* Propotional Justified Representation (PJR)

* Extended Justified Representation (EJR)

* Priceability

* The core property

Instead of using the abcvoting Python library, you can also use the 
[**abcvoting web application**](https://pref.tools/abcvoting/) by Dominik Peters
(which is based on this Python library).

## Installation

As simple as:

```bash
pip install abcvoting
```

Further details can be found [here](https://abcvoting.readthedocs.io/en/latest/installation.html).

## Development

Install all dependencies including development requirements and the abcvoting package in
[development mode](https://setuptools.readthedocs.io/en/latest/userguide/development_mode.html):

```bash
pip install -e ".[dev]"
```

Basic unit tests can be run by excluding tests which require additional dependencies:

```bash
pytest  -m "not ortools and not gmpy2 and not slow" tests/
```

For development, configure the black formatter and pre-commit hooks - see below. Also installing
all optional dependencies is recommended.

A development package is build for every commit on the master branch and uploaded to the test
instance of PyPI. It can be installed using the following command:

```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple abcvoting
```


### Black formatting

Code needs to be formatted using the [black formatter](https://black.readthedocs.io/en/stable/). This is
checked by Github actions.
[Configure your editor](https://black.readthedocs.io/en/stable/integrations/editors.html) to run the
black formatter.

### Pre-commit hooks

Pre-commit hooks are not required, but they are recommended for development.
[Pre-commit](https://pre-commit.com/) is used to manage and maintain pre-commit hooks. Install
pre-commit (e.g. via apt, conda or pip) and then run `$ pre-commit install` to install the hooks.


## References

[1] Piotr Faliszewski, Piotr Skowron, Arkadii Slinko, and Nimrod Talmon. Multiwinner voting: A
new challenge for social choice theory. In Ulle Endriss, editor, Trends in Computational Social
Choice, chapter 2, pages 27–47. AI Access, 2017. http://research.illc.uva.nl/COST-IC1205/BookDocs/Chapters/TrendsCOMSOC-02.pdf

[2] Lackner, Martin, and Piotr Skowron. "Multi-Winner Voting with Approval Preferences". 
Springer International Publishing, SpringerBriefs in Intelligent Systems , 2023.
https://link.springer.com/book/10.1007/978-3-031-09016-5

<!--
[2] Markus Brill, Rupert Freeman, Svante Janson and Martin Lackner. Phragmén's Voting Methods and Justified Representation. In Proceedings of the 31st AAAI Conference on Artificial Intelligence (AAAI 2017), pages 406-413, AAAI Press, 2017. https://arxiv.org/abs/2102.12305

[3] Steven J Brams, D Marc Kilgour, and M Remzi Sanver. A minimax procedure for electing committees. Public Choice, 132(3-4):401–420, 2007.

[4] Martin Lackner, Piotr Skowron.
A Quantitative Analysis of Multi-Winner Rules. arXiv preprint arXiv:1801.01527. 2018. https://arxiv.org/abs/1801.01527

[5] Properties of multiwinner voting rules.
Edith Elkind, Piotr Faliszewski, Piotr Skowron, and Arkadii Slinko.
Social Choice and Welfare volume 48, pages 599–632. 2017. https://link.springer.com/article/10.1007/s00355-017-1026-z

[6] Peters, Dominik, and Piotr Skowron.
Proportionality and the Limits of Welfarism. arXiv preprint arXiv:1911.11747. 2019. https://arxiv.org/abs/1911.11747

-->
