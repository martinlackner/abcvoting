---
title: 'abcvoting: A Python package for approval-based multi-winner voting rules'
tags:
  - Python
  - voting
  - committee voting
  - social choice
  - approval voting 
authors:
  - name: Martin Lackner
    orcid: 0000-0003-2170-0770
    affiliation: 1
affiliations:
 - name: TU Wien, Vienna, Austria
   index: 1
date: 15 September 2022
bibliography: paper.bib
---

# Summary

The Python package `abcvoting` is a research tool to explore and analyse
*approval-based committee (ABC) elections* [@FSST-trends; @abcbook]. 
First and foremost, it contains implementations
of major ABC voting rules. These are voting rules that accept as input
*approval ballots*, that is, the (binary) preferences of voters expressing 
which candidates they like or support.
The output is a fixed-size subset of candidates, called a *committee*.
Different ABC voting rules represent different approaches how such a committee
should be formed.
For example, there is a trade-off between selecting widely supported candidates
and selecting candidates that represent as many voters as possible [@aij/guarantees].
Much of the recent research has focussed on developing ABC voting rules
that reflect the preferences of voters in a *proportional* fashion.

`abcvoting` is primarily intended for researchers interested in voting
and related algorithmic challenges.
The core content of `abcvoting` are implementations of a large number
of ABC voting rules. These allow a user to quickly compute (and compare)
winning committees for all implemented voting rules. 
In addition to computing winning committees, `abcvoting` can be used to
verify axiomatic properties of committees. Axiomatic properties are 
mathematical formalizations of desirable features, e.g, fairness guarantees.
Such properties are fundamental to the analysis and discussion
of voting rules.

In a bit more detail, `abcvoting` has the following functionality:

- Algorithms for **computing winning committees** of many ABC voting rules,
  including 
  - Proportional Approval Voting (PAV), Chamberlin-Courant (CC), and arbitrary Thiele methods, 
  - Sequential and Reverse-Sequential Thiele methods,
  - Phragmén's sequential rule and other rules by Phragmén,
  - Monroe's rule and its approximation Greedy Monroe,
  - the Method of Equal Shares,
  - and many more.
    
  We refer to the book by [@abcbook] for an overview and explanations of these and other ABC voting rules.
- Functions for **reading and writing election (preference) data**. 
  In particular, it supports the established Preflib format [@MatteiW13].
- Functions for **generating ABC elections from probabilistic distributions**, 
  such as the Truncated Mallows distribution, Independent Culture,
  Resampling, and the Truncated Pólya Urn model (see the work of
  @howtosample for details).
- Algorithms for analyzing the **axiomatic properties** of a given committee. 
  To name a few important properties, `abcvoting`
supports Proportional Justified Representation [@pjr17],
Extended Justified Representation [@justifiedRepresentation], 
Priceability [@pet-sko:laminar], 
and the Core property [@justifiedRepresentation].

# Statement of need

In the last years, approval-based committee voting has become an increasingly active
topic of research within the artificial intelligence community
(in particular its subfield *computational social choice*).
While originally most of the research on this topic has been of theoretical nature,
more and more recent publications complement theoretical work with practical,
computional evaluations. Thus, there is a growing need for
well-tested implementations of common ABC voting rules that can serve as
a basis for experimental evaluations of new concepts.

Moreover, many computational problems related to ABC elections are computationally
difficult (NP-hard or harder). For example, many ABC voting rules are formulated
as optimization problems, where the goal is to find a committee
maximizing a certain score. As there are exponentially many possible committees,
it requires non-trivial algorithmic techniques to compute winning committees in reasonable time.
The same holds for axiomatic properties: many of these are also computationally
hard to verify.
`abcvoting` uses a number of techniques to deal with this computational complexity:
integer linear programs, branch-and-bound algorithms, constraint programming,
and others. Many voting rules are implemented via more than one algorithm.
This is useful for correctness tests and algorithmic comparisons. 

The `abcvoting` package has been used in a number of publications
 [@aij/guarantees; @godziszewski2021analysis; @howtosample; @FairsteinVMG22; @brill2022individual].
In addition, it contains Python code for many of the examples appearing in 
the book *Multi-winner voting with approval preferences* [@abcbook].

# Acknowledgements

The following people have contributed code to this package or provided help with technical 
and scientific questions (in alphabetical order):
Pawel Batko, Elvi Cela, Piotr Faliszewski, Stefan Schlomo Forster, Andrzej Kaczmarczyk, 
Jonas Kompauer, Benjamin Krenn, Florian Lackner,
Dominik Peters, Peter Regner, Piotr Skowron, Stanisław Szufa.

The development of this package was supported by the Austrian Science Fund FWF, grant P31890.

# References

