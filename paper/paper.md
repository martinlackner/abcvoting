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
*approval-based committee (ABC) elections* [@abcbook]. It contains implementations
of major ABC voting rules. These are voting rules that accept as input
*approval ballots*, where voters indicate which candidates they like or support.
The output is a fixed-size subset of candidates, called a *committee*.
Ideally, a committee is chosen in such a fashion that it reflects the input,
that is, the voters' opinion as expressed via approval ballots.

`abcvoting` is primarily designed for researchers interested in voting
and related algorithmic challenges.
The core content of `abcvoting` are implementations of a large number
of ABC voting rules. These allow a user to quickly compute (and compare)
winning committees for all implemented voting rules. 
In addition to computing winning committees, `abcvoting` can be used to
verify axiomatic properties of committees. Axiomatic properties are 
mathematical formalizations of desirable features, e.g, fairness guarantees.
Such properties are fundamentally important to the analysis and discussion
of voting rules. To name a few important axiomatic properties, `abcvoting`
supports Proportional Justified Representation [@pjr17],
Extended Justified Representation [@justifiedRepresentation], 
Priceability [@pet-sko:laminar], 
and the Core property [@justifiedRepresentation].
`abcvoting` also contains functionality to read and write election 
data, including in the established Preflib format [@MatteiW13].

# Statement of need

A main reason for the existence of `abcvoting` is that
many computational problems related to ABC elections are computationally
hard (NP-hard or harder). For example, many ABC voting rules are formulated
as optimization problems. As a consequence, it requires effort to find 
suitable solution techniques to compute winning committees.
The same holds for axiomatic properties: many of those are also computationally
hard to verify.
`abcvoting` uses a number of techniques to deal with this computational complexity:
integer linear programs, branch-and-bound algorithms, constraint programming,
and others. Many voting rules are implemented via more than one algorithm.
This is useful for correctness tests and algorithmic comparisons. 

The `abcvoting` package has been used in a number of publications
 [@godziszewski2021analysis; @howtosample; @FairsteinVMG22; @brill2022individual; @aij/guarantees].
In addition, it contains Python code for many of the examples appearing in 
the book *Multi-winner voting with approval preferences* [@abcbook].

# Acknowledgements

The following people have contributed code to this package or provided help with technical and scientific questions (in alphabetic order):
Pawel Batko, Elvi Cela, Piotr Faliszewski, Stefan Schlomo Forster, Andrzej Kaczmarczyk, Jonas Kompauer, Benjamin Krenn, Florian Lackner,
Dominik Peters, Peter Regner, Piotr Skowron, Stanisław Szufa.

The development of this package has been supported by the Austrian Science Fund FWF, grant P31890.

# References

