# abcvoting

## Python implementations of approval-based committee (multi-winner) rules

Approval-based committee rules (ABC rules) are voting methods for selecting a committee, i.e., a fixed-size subset of candidates.
ABC rules are also known as approval-based multi-winner rules.
The input of such rules are [approval ballots](https://en.wikipedia.org/wiki/Approval_voting#/media/File:Approval_ballot.svg).
We recommend the survey by Faliszewski et al. [1] as an introduction to this topic and for further reference.
The following ABC rules are implemented:

* Approval Voting (AV)

* Satisfaction Approval Voting (SAV)

* Proportional Approval Voting (PAV) [Gurobi optional]

* Sequential Proportional Approval Voting

* Reverse Sequential Proportional Approval Voting

* Sainte-Lagu&euml; Approval Voting (SLAV) [4]

* Sequential Sainte-Lagu&euml; Approval Voting [4]

* Approval Chamberlin-Courant (CC) [Gurobi optional]

* Sequential Chamberlin-Courant

* Reverse Sequential Chamberlin-Courant

* Phragmen's sequential rule (see [2])
  
* Monroe [Gurobi optional]

* Minimax Approval Voting (see [3])

* Greedy Monroe

* Rule X

## Example

The following code computes the Proportional Approval Voting (PAV) rule for a profile with 6 voters and 5 candidates.

```python
from preferences import Profile
import rules_approval

profile = Profile(5)
profile.add_preferences([[0,1,2], [0,1], [0,1], [1,2], [3,4], [3,4]])
committeesize = 3
print rules_approval.compute_pav(profile, committeesize, ilp=False)
```
The output is 
```
[[0, 1, 3], [0, 1, 4]]
```
which corresponds to the two committees {0,1,3} and {0,1,4}. Further examples can be found in [examples/examples.py](examples/examples.py), [examples/file_examples.py](examples/file_examples.py) and [examples/random_profile_examples.py](examples/random_profile_examples.py).

## Comments

* This module requires Python 2.7+ or 3.6+.
* The used modules can be found in [requirements.txt](requirements.txt), the two following are not always required.
* Most computationally hard rules are also implemented via the ILP solver [Gurobi](http://www.gurobi.com/). The corresponding functions require [gurobipy](https://www.gurobi.com/documentation/8.1/quickstart_mac/the_gurobi_python_interfac.html).
* Some functions use fractions (e.g., `compute_seqphragmen`). These compute significantly faster if the module [gmpy2](https://gmpy2.readthedocs.io/) is available. If gmpy2 is not available, the much slower Python module [fractions](https://docs.python.org/2/library/fractions.html) is used.
* All voting methods have a parameter `resolute`. If it is set to true, only one winning committee is computed. In most cases,  `resolute=True` speeds up the computation. 
* For ILP implementations via Gurobi it is not guaranteed that all winning committees are computed even if `resolute = False`.


## Contributors

The following people have contributed code to this package and provided help with technical and scientific questions (in alphabetic order):
[Piotr Faliszewski](http://home.agh.edu.pl/~faliszew/),
[Stefan Schlomo Forster](https://github.com/stefanschlomoforster),
[Andrzej Kaczmarczyk](http://www.user.tu-berlin.de/droores/),
[Martin Lackner](http://martin.lackner.xyz/),
[Dominik Peters](http://dominik-peters.de/), 
[Piotr Skowron](https://www.mimuw.edu.pl/~ps219737/).


## References

[1] Piotr Faliszewski, Piotr Skowron, Arkadii Slinko, and Nimrod Talmon. Multiwinner voting: A
new challenge for social choice theory. In Ulle Endriss, editor, Trends in Computational Social
Choice, chapter 2, pages 27–47. AI Access, 2017. http://research.illc.uva.nl/COST-IC1205/BookDocs/Chapters/TrendsCOMSOC-02.pdf

[2] Markus Brill, Rupert Freeman, Svante Janson and Martin Lackner. Phragmén's Voting Methods and Justified Representation. In Proceedings of the 31st AAAI Conference on Artificial Intelligence (AAAI 2017), pages 406-413, AAAI Press, 2017. http://martin.lackner.xyz/publications/phragmen.pdf

[3] Steven J Brams, D Marc Kilgour, and M Remzi Sanver. A minimax procedure for electing committees. Public Choice, 132(3-4):401–420, 2007.

[4] Martin Lackner, Piotr Skowron.
A Quantitative Analysis of Multi-Winner Rules. CoRR abs/1801.01527. 2018. https://arxiv.org/abs/1801.01527

