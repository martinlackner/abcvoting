A brief introduction
====================

A simple example
----------------

The following code computes the Proportional Approval Voting (PAV) rule for an approval profile.
Let's first create the approval profile for (up to) five candidates.
These candidates correspond to the numbers 0 to 4.

.. doctest::

    >>> from abcvoting.preferences import Profile
    >>> from abcvoting import abcrules

    >>> profile = Profile(num_cand=5)

Then we add six voters, specified by the candidates that they approve.
The first voter approves candidates 0, 1, and 2,
the second voter approves candidate 3, etc.

.. doctest::

    >>> profile.add_voters([{0, 1, 2}, {3}, {0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1}, {4}])

And finally, let's compute the winning committees for this profile according to PAV for a committee size of 3.

.. doctest::

    >>> print(abcrules.compute_pav(profile, committeesize=3))
    [{0, 1, 3}, {0, 1, 4}]

We see that there are two winning committees: {0,1,3} and {0,1,4}.

We can also compute the winning committees for several ABC voting rules at once.
Let's compute them for Approval Voting (AV), Sequential Approval Chamberlin-Courant (seq-CC), and
Phragmén's Sequential Rule (seq-Phragmén).

These ABC rules are identified by their `rule_id`: `"av"`, `"seqcc"`, and `"seqphragmen"`, respectively.

.. doctest::

    >>> for rule_id in ["av", "seqcc", "seqphragmen"]:
    ...    print(abcrules.compute(rule_id, profile, committeesize=3))
    [{0, 1, 2}]
    [{0, 3, 4}]
    [{0, 1, 3}]

Each of the three rules yields a different winning committee.
