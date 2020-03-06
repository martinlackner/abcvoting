# abcvoting

## Python implementations of approval-based committee (multi-winner) rules

Approval-based committee rules (ABC rules) are voting methods for selecting a committee, i.e., a fixed-size subset of candidates.
ABC rules are also known as approval-based multi-winner rules.
The input of such rules are [approval ballots](https://en.wikipedia.org/wiki/Approval_voting#/media/File:Approval_ballot.svg).
We recommend the survey by Faliszewski et al. [1] as an introduction to this topic and for further reference.
The following ABC rules are implemented:

* Approval Voting (AV)

* Satisfaction Approval Voting (SAV)

* Proportional Approval Voting (PAV)

* Sequential Proportional Approval Voting (seq-PAV)

* Reverse Sequential Proportional Approval Voting (revseq-PAV)

* Sainte-Lagu&euml; Approval Voting (SLAV) [4]

* Sequential Sainte-Lagu&euml; Approval Voting [4]

* Approval Chamberlin-Courant (CC)

* Sequential Chamberlin-Courant

* Reverse Sequential Chamberlin-Courant

* Phragmén's sequential rule (see [2])
  
* Monroe's rule

* Minimax Approval Voting (see [3])

* Greedy Monroe (see [5])

* Rule X (see [6])

* Phragmén's First Method (Enestr&ouml;m's Method) [7]

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
which corresponds to the two committees {0,1,3} and {0,1,4}. Further examples can be found in the directory [examples/](examples/).

## Comments

* This module requires Python 2.7+ or 3.6+.
* The used modules can be found in [requirements.txt](requirements.txt).
* Most computationally hard rules are also implemented via the ILP solver [Gurobi](http://www.gurobi.com/). The corresponding functions require [gurobipy](https://www.gurobi.com/documentation/8.1/quickstart_mac/the_gurobi_python_interfac.html).
* Some functions use fractions (e.g., `compute_seqphragmen`). These compute significantly faster if the module [gmpy2](https://gmpy2.readthedocs.io/) is available. If gmpy2 is not available, the much slower Python module [fractions](https://docs.python.org/2/library/fractions.html) is used.
* All voting methods have a parameter `resolute`. If it is set to true, only one winning committee is computed. In most cases,  `resolute=True` speeds up the computation. 
* For ILP implementations via Gurobi it is not guaranteed that all winning committees are computed even if `resolute = False`.


## Acknowledgements

The following people have contributed code to this package or provided help with technical and scientific questions (in alphabetic order):
[Piotr Faliszewski](http://home.agh.edu.pl/~faliszew/),
[Stefan Schlomo Forster](https://github.com/stefanschlomoforster),
[Andrzej Kaczmarczyk](http://www.user.tu-berlin.de/droores/),
[Benjamin Krenn](https://github.com/benjaminkrenn),
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
A Quantitative Analysis of Multi-Winner Rules. arXiv preprint arXiv:1801.01527. 2018. https://arxiv.org/abs/1801.01527

[5] Properties of multiwinner voting rules.
Edith Elkind, Piotr Faliszewski, Piotr Skowron, and Arkadii Slinko. 
Social Choice and Welfare volume 48, pages 599–632. 2017. https://link.springer.com/article/10.1007/s00355-017-1026-z

[6] Peters, Dominik, and Piotr Skowron. 
Proportionality and the Limits of Welfarism. arXiv preprint arXiv:1911.11747. 2019. https://arxiv.org/abs/1911.11747

[7] Janson, Svante.
Phragmén's and Thiele's election methods. arXiv preprint arXiv:1611.08826. 2016. https://arxiv.org/pdf/1611.08826.pdf
