ABC rules
=========

Approval-based committee rules (ABC rules) are voting methods for selecting a committee,
i.e., a fixed-size subset of candidates.
ABC rules are also known as approval-based multi-winner rules.
The input of such rules are
`approval ballots
<https://en.wikipedia.org/wiki/Approval_voting>`_.
We recommend the book
`Multi-Winner Voting with Approval Preferences <https://arxiv.org/abs/2007.01795>`_ [1]_
by Lackner and Skowron as a detailed introduction
to ABC rules and related research directions.
In addition, the
`survey by Faliszewski et al. <http://research.illc.uva.nl/COST-IC1205/BookDocs/Chapters/TrendsCOMSOC-02.pdf>`_ [2]_
is useful as a more general introduction to committee voting (not limited to approval ballots).

The main ABC rules implemented in `abcvoting` are the following:

.. testsetup::

    from abcvoting import abcrules

.. doctest::

    >>> for rule_id in abcrules.MAIN_RULE_IDS:
    ...     print(f"{rule_id:20s}{abcrules.get_rule(rule_id).longname}")
    av                  Approval Voting (AV)
    sav                 Satisfaction Approval Voting (SAV)
    pav                 Proportional Approval Voting (PAV)
    slav                Sainte-Laguë Approval Voting (SLAV)
    cc                  Approval Chamberlin-Courant (CC)
    lexcc               Lexicographic Chamberlin-Courant (lex-CC)
    geom2               2-Geometric Rule
    seqpav              Sequential Proportional Approval Voting (seq-PAV)
    revseqpav           Reverse Sequential Proportional Approval Voting (revseq-PAV)
    seqslav             Sequential Sainte-Laguë Approval Voting (seq-SLAV)
    seqcc               Sequential Approval Chamberlin-Courant (seq-CC)
    seqphragmen         Phragmén's Sequential Rule (seq-Phragmén)
    minimaxphragmen     Phragmén's Minimax Rule (minimax-Phragmén)
    leximinphragmen     Phragmén's Leximin Rule (leximin-Phragmén)
    monroe              Monroe's Approval Rule (Monroe)
    greedy-monroe       Greedy Monroe
    minimaxav           Minimax Approval Voting (MAV)
    lexminimaxav        Lexicographic Minimax Approval Voting (lex-MAV)
    rule-x              Rule X (aka Method of Equal Shares)
    phragmen-enestroem  Method of Phragmén-Eneström
    consensus-rule      Consensus Rule
    trivial             Trivial Rule
    rsd                 Random Serial Dictator

The short identifiers on the left side are the respective `rule_id`'s.

In addition to these rules, there are Thiele methods, Sequential Thiele methods,
and Reverse Sequential Thiele methods. These are classes of ABC rules and
are parameterized by arbitary scoring functions (explained in detail here [1]_).
For example, the 3-Geomtric Thiele method (with `rule_id` being `"geom3"`) as well as
Sequential 3-Geometric (`rule_id="seqgeom3"`) and Reverse Sequential 3-Geometric (`rule_id="revseqgeom3"`)
are implemented but not specifically listed above.

.. doctest::

    >>> for rule_id in ["geom3", "seqgeom3", "revseqgeom3"]:
    ...     print(f"{rule_id:20s}{abcrules.get_rule(rule_id).longname}")
    geom3               3-Geometric Rule
    seqgeom3            Sequential 3-Geometric Rule
    revseqgeom3         Reverse Sequential 3-Geometric Rule


.. [1] Lackner, Martin, and Piotr Skowron.
    "Multi-Winner Voting with Approval Preferences".
    arXiv preprint arXiv:2007.01795. 2020.
    `<https://arxiv.org/abs/2007.01795>`_

.. [2] Piotr Faliszewski, Piotr Skowron, Arkadii Slinko, and Nimrod Talmon. Multiwinner voting: A
    new challenge for social choice theory. In Ulle Endriss, editor, Trends in Computational Social
    Choice, chapter 2, pages 27–47. AI Access, 2017.
    http://research.illc.uva.nl/COST-IC1205/BookDocs/Chapters/TrendsCOMSOC-02.pdf
