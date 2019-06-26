# approval-multiwinner

## Python implementations of approval-based multi-winner rules

Approval-based multi-winner rules are voting methods for selecting a committee, i.e., a fixed-size subset of candidates. We recommend the suvey by Faliszewski et al. [1] as an introduction to this topic and for further reference.
The following approval-based multi-winner rules are implemented:

* Approval Voting (AV)

* Satisfaction Approval Voting (SAV)

* Proportional Approval Voting (PAV) [Gurobi optional]

* Sequential Proportional Approval Voting (seq-PAV)

* Reverse Sequential Proportional Approval Voting (revseq-PAV)

* Approval Chamberlin-Courant (CC) [Gurobi optional]

* Sequential Chamberlin-Courant (seq-CC)

* Reverse Sequential Chamberlin-Courant (revseq-CC)

* Phragmen's sequential rule (see [2])
  
* Monroe [Gurobi optional]

* Maximin Approval Voting 

## Example

The following code computes the Proportional Approval Voting (PAV) rule for a profile with 6 voters and 5 candidates.

```python
from preferences import Profile
import rules_approval

profile = Profile(5)
profile.add_preferences([[0,1,2],[0,1],[0,1],[1,2],[3,4],[3,4]])
committeesize = 3
print rules_approval.compute_pav(profile,committeesize,ilp=False)
```
The output is 
```
[[0, 1, 3], [0, 1, 4]]
```
which corresponds to the two committees {0,1,3} and {0,1,4}. Further examples can be found in [examples.py](examples.py).

## Comments

* Most computationally hard rules are also implemented via the ILP solver [Gurobi](http://www.gurobi.com/). These methods require [gurobipy](https://www.gurobi.com/documentation/8.1/quickstart_mac/the_gurobi_python_interfac.html).
* All voting methods have a parameter `resolute`. If it set to true, only one winning committee is computed.
* For ILP implementations via Gurobi it is not guaranteed that all winning committees are computed even if `resolute = False`.


## Acknowledgements

Piotr Faliszewski, Andrzej Kaczmarczyk, Dominik Peters, and Piotr Skowron have contributed code to this package and provided help with technical and scientific questions. The [Hopcroft-Karp bipartite matching algorithm](bipartite_matching) is part of the [ActiveState Code Recipes](https://github.com/ActiveState/code) and was implemented by David Eppstein.

## References

[1] Piotr Faliszewski, Piotr Skowron, Arkadii Slinko, and Nimrod Talmon. Multiwinner voting: A
new challenge for social choice theory. In Ulle Endriss, editor, Trends in Computational Social
Choice, chapter 2, pages 27–47. AI Access, 2017. http://research.illc.uva.nl/COST-IC1205/BookDocs/Chapters/TrendsCOMSOC-02.pdf

[2] Markus Brill, Rupert Freeman, Svante Janson and Martin Lackner. Phragmén's Voting Methods and Justified Representation. In Proceedings of the 31st AAAI Conference on Artificial Intelligence (AAAI 2017), pages 406-413, AAAI Press, 2017. http://martin.lackner.xyz/publications/phragmen.pdf
