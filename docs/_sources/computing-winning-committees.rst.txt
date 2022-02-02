Computing winning committees
============================

One of the main goals of `abcvoting` is to provide a simple way to compute winning committees for a given
ABC rule. Here is a bit more detailed explanation of how to do this. (A first, simple explanation can be found
:doc:`here <simple-example>`.

First steps
-----------

ABC rules are identified by an identifier called `rule_id`. :doc:`Here <intro-abcrules>` is a list of these
identifiers. Say we are interested in Proportional Approval Voting, short: PAV. Its `rule_id` is `"pav"`.

The input of an ABC rule consists of a profile and a desired committee size. Once we know the `rule_id` and
have a profile and a desired committee size, we can compute the ABC rule.
In the following, we use the simple example used :doc:`previously <intro-abcrules>`.

.. doctest::

    >>> from abcvoting.preferences import Profile
    >>> from abcvoting import abcrules

    >>> rule_id = "pav"
    >>> profile = Profile(num_cand=5)
    >>> profile.add_voters([{0, 1, 2}, {3}, {0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1}, {4}])
    >>> print(abcrules.compute("pav", profile, committeesize=3))
    [{0, 1, 3}, {0, 1, 4}]

Number of winning committees
----------------------------

Most ABC voting rules may return more than one winning committees; these committees are "tied".
Sometimes one is only interested in one (or a just few) winning committees.
As it is generally computationally more difficult to compute all winning committees, also runtime considerations
may necessitate a limit.

There are two ways how to determine the maximum number of winning committees: the parameter `resolute`
and the parameter `max_num_of_committees`. Both are implemented for the `compute` functions of all ABC rules.

If `resolute=True`, only one winning committee is returned.

.. doctest::

    >>> print(abcrules.compute("pav", profile, committeesize=3, resolute=True))
    [{0, 1, 3}]

If `resolute=True`, the maximum number of winning committees is determined by `max_num_of_committees`.

.. doctest::

    >>> print(abcrules.compute("pav", profile, committeesize=3, max_num_of_committees=1))
    [{0, 1, 3}]

.. doctest::

    >>> print(abcrules.compute("pav", profile, committeesize=3, max_num_of_committees=4))
    [{0, 1, 3}, {0, 1, 4}]

While most ABC rule are implemented for both  `resolute=True` and `resolute=False`, for some one choice is
more natural than the other.
The default value for `resolute` is chosen to reflect this.
For example,

.. doctest::

    >>> abcrules.get_rule("pav").resolute_values
    [False, True]

The first entry in this list is the default choice. That is, if we do not provide the `resolute` parameter
when computing PAV, all winning committees are computed (`resolute=False`).
For sequential rules (such as Sequential PAV and Reverse Sequential PAV), the default choice is `resolute=True`.

Finally, the default choice of `max_num_of_committees` is

.. doctest::

    >>> print(abcrules.MAX_NUM_OF_COMMITTEES_DEFAULT)
    None

i.e., when `resolute=False`, indeed all winning committees are computed.

.. important::

    Note that `max_num_of_committees=None` (i.e., an unrestricted maximum number of winning committees)
    can lead to runtime and memory problems when there is a huge number of winning committees.

Algorithms
----------

Most ABC rules can be computed with several algorithms. For example, for PAV, we have

.. doctest::

    >>> print(abcrules.get_rule("pav").algorithms)
    ('gurobi', 'mip-gurobi', 'mip-cbc', 'branch-and-bound', 'brute-force')

These algorithms are sorted by speed (in approximation). By default, ABC rules are computed with
`algorithm="fastest"`, which picks the first available algorithm in this list.

Not all algorithms are necessarily available as some of them have optional dependencies.
Let us briefly discuss these.

Throughout `abcvoting`, the following kinds of algorithms are used:

.. doctest::

    >>> for algo_id, description in abcrules.ALGORITHM_NAMES.items():
    ...     print(f"{algo_id:20s} : {description}")
    gurobi               : Gurobi ILP solver
    branch-and-bound     : branch-and-bound
    brute-force          : brute-force
    mip-cbc              : CBC ILP solver via Python MIP library
    mip-gurobi           : Gurobi ILP solver via Python MIP library
    standard             : Standard algorithm
    standard-fractions   : Standard algorithm (using standard Python fractions)
    gmpy2-fractions      : Standard algorithm (using gmpy2 fractions)
    float-fractions      : Standard algorithm (using floats instead of fractions)
    ortools-cp           : OR-Tools CP-SAT solver

In addition to the dependencies of abcvoting [#]_, some algorithms have additional requirements:

- `gurobi` and `mip-gurobi` require Gurobi
  (`installation <https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python->`_)

- `gmpy2-fractions` requires the Python module `gmpy2`.

- All other algorithms work "out of the box".


.. [#] If `abcvoting` is installed via
    ``pip install abcvoting``, then all dependencies are installed automatically. If `abcvoting` is installed
    from source, run ``python setup.py install``.
