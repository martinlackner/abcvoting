[![DOI](https://zenodo.org/badge/192713860.svg)](https://zenodo.org/badge/latestdoi/192713860)
[![MIT License](https://badgen.net/github/license/martinlackner/abcvoting)](https://choosealicense.com/licenses/mit/)
[![PyPi](https://badgen.net/pypi/v/abcvoting)](https://pypi.org/project/abcvoting/)
![Python versions](https://badgen.net/pypi/python/abcvoting)
[![Build badge](https://github.com/martinlackner/abcvoting/workflows/Build/badge.svg)](https://github.com/martinlackner/abcvoting/actions)
[![Unittests badge](https://github.com/martinlackner/abcvoting/workflows/Unittests/badge.svg)](https://github.com/martinlackner/abcvoting/actions)
[![codecov](https://codecov.io/gh/martinlackner/abcvoting/branch/master/graph/badge.svg)](https://codecov.io/gh/martinlackner/abcvoting)

# abcvoting

## Python implementations of approval-based committee (multi-winner) rules

Approval-based committee rules (ABC rules) are voting methods for selecting a committee, i.e., a fixed-size subset of candidates.
ABC rules are also known as approval-based multi-winner rules.
The input of such rules are [approval ballots](https://en.wikipedia.org/wiki/Approval_voting#/media/File:Approval_ballot.svg).
We recommend [''Approval-Based Committee Voting''](https://arxiv.org/abs/2007.01795) by Lackner and Skowron as a detailed introduction to ABC rules and related research directions [2].
In addition, the [survey by Faliszewski et al.](http://research.illc.uva.nl/COST-IC1205/BookDocs/Chapters/TrendsCOMSOC-02.pdf) [1] is useful as a more general introduction to committee voting (not limited to approval ballots).

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

* Rule X

* Phragmén's First Method (Enestr&ouml;m's Method)

* and many more ...

## Example

The following code computes the Proportional Approval Voting (PAV) rule for a profile with 6 voters and 5 candidates. Candidates correspond to the numbers 0 to 4.

```python
from abcvoting.preferences import Profile
from abcvoting import abcrules

# a preference profile with 5 candidates (0, 1, 2, 3, 4)
profile = Profile(5)

# add six voters, specified by the candidates that they approve;
# the first voter approves candidates 0, 1, and 2,
# the second voter approves candidates 0 and 1, etc.
profile.add_voters([{0,1,2}, {0,1}, {0,1}, {1,2}, {3,4}, {3,4}])
committeesize = 3

# find winning committees for this profile according to PAV
print(abcrules.compute_pav(profile, committeesize))
```
The output is 
```
[{0, 1, 3}, {0, 1, 4}]
```
which corresponds to the two winning committees {0,1,3} and {0,1,4}. Further examples can be found in the directory [examples/](examples).
In [examples/survey/](examples/abcsurvey), all examples from the survey on ABC rules [2] are implemented. 

## Usage

At the moment there is no command line interface. The package can be used only as Python model as
shown in the example above.

Notes:

* Most computationally hard rules are also implemented via the ILP solver [Gurobi](http://www.gurobi.com/). The corresponding functions require [gurobipy](https://www.gurobi.com/documentation/8.1/quickstart_mac/the_gurobi_python_interfac.html).
* Some functions use fractions (e.g., `compute_seqphragmen`). These compute significantly faster if the module [gmpy2](https://gmpy2.readthedocs.io/) is available. If gmpy2 is not available, the much slower Python module [fractions](https://docs.python.org/2/library/fractions.html) is used.
* All voting methods have a parameter `resolute`. If it is set to true, only one winning committee is computed. In most cases, `resolute=True` speeds up the computation. 

## Installation

Using pip:

```bash
pip install abcvoting
```

Latest development version from source:

```bash
git clone https://github.com/martinlackner/abcvoting/
python setup.py install
```

Requirements:
* Python 3.6+
* see [setup.py](setup.py) for 3rd party dependencies

Optional requirements:
* gmpy2
* cvxpy
* solvers:
  * Gurobi
  * GLPK_MI
  * CBC
  * Scip

<!-- TODO: add instructions for installation of solvers -->

## How to Cite

If you would like to cite abcvoting in a research paper or text,
please use the following (or a similar) citation:

```
M. Lackner, P. Regner, B. Krenn, and S. S. Forster.
abcvoting: A Python library of approval-based committee voting rules, 2021.
URL https://doi.org/10.5281/zenodo.3904466.
Current version: https://github.com/martinlackner/abcvoting.
```

Bibtex:

```
@misc{abcvoting,
  author       = {Martin Lackner and
                  Peter Regner and
                  Benjamin Krenn and
                  Stefan Schlomo Forster},
  title        = {{abcvoting: A Python library of approval-based 
                   committee voting rules}},
  year         = 2021,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3904466},
  url          = {https://doi.org/10.5281/zenodo.3904466},
  note         = {Current version: \url{https://github.com/martinlackner/abcvoting}}
}
```


## Development

Install all dependencies including development requirements and the abcvoting package in
[development mode](https://setuptools.readthedocs.io/en/latest/userguide/development_mode.html):

```bash
pip install -e .[dev]
```

Basic unit tests can be run by excluding tests which require additional dependencies:

```bash
pytest  -m "not gurobi and not scip and not cbc and not glpk_mi and not cvxpy and not gmpy2 and not slow" tests/
```

For development, configure the black formatter and pre-commit hooks - see below. Also installing
all optional dependencies is recommended.


### Black formatting

Code needs to be formatted using the [black formatter](https://black.readthedocs.io/en/). This is
checked by Github actions.
[Configure your editor](https://black.readthedocs.io/en/latest/editor_integration.html), to run the
black formatter.

### Pre-commit hooks

Pre-commit hooks are not required, but they are recommended for development.
[Pre-commit](https://pre-commit.com/) is used to manage and maintain pre-commit hooks. Install
pre-commit (e.g. via apt, conda or pip) and then run`$ pre-commit install` to install the hooks.

## Acknowledgements

The following people have contributed code to this package or provided help with technical and scientific questions (in alphabetic order):
[Piotr Faliszewski](http://home.agh.edu.pl/~faliszew/),
[Stefan Schlomo Forster](https://github.com/stefanschlomoforster),
[Andrzej Kaczmarczyk](http://www.user.tu-berlin.de/droores/),
[Benjamin Krenn](https://github.com/benjaminkrenn),
[Martin Lackner](http://martin.lackner.xyz/),
[Pawel Batko](https://github.com/pbatko),
[Dominik Peters](http://dominik-peters.de/),
[Peter Regner](https://github.com/lumbric), 
[Piotr Skowron](https://www.mimuw.edu.pl/~ps219737/).

The development of this module has been supported by the Austrian Science Fund FWF, grant P31890.

## References

[1] Piotr Faliszewski, Piotr Skowron, Arkadii Slinko, and Nimrod Talmon. Multiwinner voting: A
new challenge for social choice theory. In Ulle Endriss, editor, Trends in Computational Social
Choice, chapter 2, pages 27–47. AI Access, 2017. http://research.illc.uva.nl/COST-IC1205/BookDocs/Chapters/TrendsCOMSOC-02.pdf

[2] Lackner, Martin, and Piotr Skowron. "Approval-Based Committee Voting: Axioms, Algorithms, and Applications." arXiv preprint arXiv:2007.01795. 2020. https://arxiv.org/abs/2007.01795


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