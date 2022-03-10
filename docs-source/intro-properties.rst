Properties of committees
========================

ABC rules return committees and these committees may or may not have certain properties.
For an overview of the many properties of committees, we refer to the survey by Lackner and Skowron [1]_.
Here, we see how one can test a given committee and find out which properties it satisfies.

.. testsetup::

    from abcvoting import abcrules
    abcrules.available_algorithms = ["brute-force", "standard", "standard-fractions"]

The following example (from [2]_) shows that Phragmén's Sequential Rule (seq-Phragmén) may output committees that fail
Extended Justified Representation (EJR) [3]_.
We first compute the winning committee ...

.. doctest::

    >>> from abcvoting.preferences import Profile
    >>> from abcvoting import abcrules, properties
    >>> from abcvoting.output import output, INFO

    >>> profile = Profile(num_cand=14)
    >>> profile.add_voters(
    ...       [{0, 1, 2}] * 2
    ...     + [{0, 1, 3}] * 2
    ...     + [{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}] * 6
    ...     + [{3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}] * 5
    ...     + [{4, 5, 6, 7, 8, 9, 10, 11, 12, 13}] * 9
    ... )
    >>> committees = abcrules.compute_seqphragmen(profile, committeesize=12)

... and then analyze this committee. We set the verbosity to `INFO` so that the result of `properties.full_analysis`
is printed.

.. doctest::

    >>> output.set_verbosity(INFO)
    >>> properties.full_analysis(profile, committees[0])
    Pareto optimality                                  : True
    Justified representation (JR)                      : True
    Proportional justified representation (PJR)        : True
    Extended justified representation (EJR)            : False

In contrast, committees returned by seq-Phragmén always satisfy JR and PJR [2]_.
Pareto optimality is not necessarily satisfied by seq-Phragmén, but it is satisfied in this instance.

.. testcleanup::

    from abcvoting.output import WARNING
    output.set_verbosity(WARNING)

.. [1] Lackner, Martin, and Piotr Skowron.
    "Multi-Winner Voting with Approval Preferences". arXiv preprint arXiv:2007.01795. 2020.
    `<https://arxiv.org/abs/2007.01795>`_

.. [2] Markus Brill, Rupert Freeman, Svante Janson and Martin Lackner.
    Phragmén's Voting Methods and Justified Representation.
    In Proceedings of the 31st AAAI Conference on Artificial Intelligence (AAAI 2017), pages 406-413, AAAI Press, 2017.
    https://arxiv.org/abs/2102.12305

.. [3] Aziz, H., Brill, M., Conitzer, V., Elkind, E., Freeman, R., & Walsh, T. (2017).
    Justified representation in approval-based committee voting.
    Social Choice and Welfare, 48(2), 461-485.
    https://arxiv.org/abs/1407.8269
