"""
Miscellaneous functions for committees (i.e., subsets of candidates).
"""

import math
import numpy as np
from time import perf_counter
import itertools

FLOAT_ISCLOSE_REL_TOL = 1e-12
"""
The relative tolerance when comparing floats.

See also: `math.isclose() <https://docs.python.org/3/library/math.html#math.isclose>`_.
"""

FLOAT_ISCLOSE_ABS_TOL = 1e-12
"""
The absolute tolerance when comparing floats.

See also: `math.isclose() <https://docs.python.org/3/library/math.html#math.isclose>`_.
"""


class CandidateSet(set):
    """
    A set of candidates, that is, a set of positive integers.

    Parameters
    ----------
        candidates : iterable
            An iterable of candidates (positive integers).

        num_cand : int, optional
            The maximum number of candidates. Used only for checks.

            If this `num_cand` is provided, it is verified that `approved` does not contain
            numbers `>= num_cand`.
    """

    def __init__(self, candidates=(), num_cand=None):
        # note: empty approval sets are fine

        super().__init__(candidates)
        if len(candidates) != len(self):
            raise ValueError(f"CandidateSet initialized with duplicate elements ({candidates}).")

        for cand in candidates:
            if not isinstance(cand, (int, np.integer)):
                raise TypeError(
                    f"Object of type {str(type(cand))} not suitable as candidate, "
                    f"only positive integers allowed."
                )

        if not all(cand >= 0 for cand in candidates):
            raise ValueError(
                f"CandidateSet initialized with elements that are not positive "
                f"integers ({candidates})."
            )

        if num_cand is not None and any(cand >= num_cand for cand in candidates):
            raise ValueError(
                f"CandidateSet initialized with elements that are >= num_cand ({num_cand}), "
                f"the number of candidate ({candidates})."
            )

    def __str__(self):
        return self.str_with_names()

    def str_with_names(self, cand_names=None):
        """
        Format a CandidateSet, using the names of candidates (instead of indices) if provided.

        Parameters
        ----------
            cand_names : list of str or str, optional
                List of symbolic names for every candidate.

        Returns
        -------
            str
        """
        return str_set_of_candidates(self, cand_names)


def sorted_committees(committees):
    """
    Sort a list of committees and ensure that committees are sets.

    Parameters
    ----------
        committees : iterable of iterable
            An iterable of committees; committees can be sets, tuples, lists, etc.

    Returns
    -------
        list of CandidateSet
            A sorted list of committees.
    """
    return sorted((CandidateSet(committee) for committee in committees), key=str)


def str_set_of_candidates(candset, cand_names=None):
    """
    Nicely format a set of candidates.

    .. doctest::

        >>> print(str_set_of_candidates({0, 1, 3, 2}))
        {0, 1, 2, 3}
        >>> print(str_set_of_candidates({0, 3, 1}, cand_names="abcde"))
        {a, b, d}

    Parameters
    ----------
        candset : iterable of int
            An iteratble of candidates.

        cand_names : list of str or str, optional
            List of symbolic names for every candidate.

    Returns
    -------
        str
    """
    if cand_names is None:
        named = sorted(str(cand) for cand in candset)
    else:
        named = sorted(str(cand_names[cand]) for cand in candset)
    return "{" + ", ".join(named) + "}"


def str_sets_of_candidates(sets_of_candidates, cand_names=None):
    """
    Nicely format a list of sets of candidates.

    .. doctest::

        >>> comm1 = CandidateSet({0, 1, 3})
        >>> comm2 = CandidateSet({0, 1, 4})
        >>> print(str_sets_of_candidates([comm1, comm2]))
         {0, 1, 3}
         {0, 1, 4}
        <BLANKLINE>
        >>> print(str_sets_of_candidates([comm1, comm2], cand_names="abcde"))
         {a, b, d}
         {a, b, e}
        <BLANKLINE>

    Parameters
    ----------
        sets_of_candidates : list of iterable of int
            A list of iterables that contain candidates (i.e., non-negative integers).

        cand_names : list of str or str, optional
            List of symbolic names for every candidate.

    Returns
    -------
        str
    """
    str_sets = [str_set_of_candidates(candset, cand_names) for candset in sets_of_candidates]
    return " " + "\n ".join(str_sets) + "\n"


def str_committees_with_header(committees, cand_names=None, winning=False):
    """
    Nicely format a list of committees including a header (stating the number of committees).

    .. doctest::

        >>> comm1 = CandidateSet({0, 1, 3})
        >>> comm2 = CandidateSet({0, 1, 4})
        >>> print(str_committees_with_header([comm1, comm2], winning=True))
        2 winning committees:
         {0, 1, 3}
         {0, 1, 4}
        <BLANKLINE>
        >>> print(str_committees_with_header([comm1, comm2], cand_names="abcde"))
        2 committees:
         {a, b, d}
         {a, b, e}
        <BLANKLINE>

    Parameters
    ----------
        committees : list of iterable of int
            A list of committees (set of positive integers).

        cand_names : list of str or str, optional
            List of symbolic names for every candidate.

        winning : bool, optional
            Write "winning committee" instead of "committee".

    Returns
    -------
        str
    """
    output = ""
    if committees is None or len(committees) < 1:
        if winning:
            return "No winning committees"
        return "No committees"
    if winning:
        commstring = "winning committee"
    else:
        commstring = "committee"
    if len(committees) == 1:
        output += "1 " + commstring + ":\n"
    else:
        output += str(len(committees)) + " " + commstring + "s:\n"
    output += str_sets_of_candidates(committees, cand_names=cand_names)
    return output


def hamming(set1, set2):
    """
    Hamming distance between sets `set1` and `set2`.

    The Hamming distance for sets is the size of their symmetric difference,
    or, equivalently, the usual Hamming distance when sets are viewed as 0-1-strings.

    Parameters
    ----------
        set1, set2 : set of int
            The two sets for which the Hamming distance is computed.

    Returns
    -------
        int
            The Hamming distance.
    """
    diffs = [x for x in set1 if x not in set2] + [x for x in set2 if x not in set1]
    return len(diffs)


def header(text, symbol="-"):
    """
    Format a header for `text`.

    Parameters
    ----------
        text : str
            Header text.

        symbol : str
            Symbol to be used for the box around the header text; should be exactly 1 character.

    Returns
    -------
        str
    """
    border = symbol[0] * len(text) + "\n"
    return border + text + "\n" + border


def powerset(iterable, max_size=None):
    """
    Yield all possible subsets of the iterable (or all subsets with at most `max-size` elements).

    From: https://docs.python.org/3/library/itertools.html#itertools-recipes

    Parameters
    ----------
        iterable : iterable
            An iterable.

        max_size : int, optional
            Maximum size of the subsets.

    Returns
    -------
        iterable
    """

    s = list(iterable)
    if max_size is None:
        max_size = len(s)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(max_size + 1))


def compare_list_of_committees(committees1, committees2):
    """
    Check whether two lists of committees are equal.

    The order of candidates and their multiplicities in these lists are ignored.
    To be precise, two lists are equal if every committee in list1 is contained in list2 and
    vice versa.
    Committees are, as usual, of type `CandidateSet` (i.e., sets of positive integers).

    .. doctest::

        >>> comm1 = CandidateSet({0, 1, 3})
        >>> comm2 = CandidateSet({0, 3, 1})  # the same set as `comm1`
        >>> comm3 = CandidateSet({0, 1, 4})
        >>> compare_list_of_committees([comm1, comm2, comm3], [comm1, comm3])
        True
        >>> compare_list_of_committees([comm1, comm3], [comm1])
        False

    Parameters
    ----------
        committees1, committees2 : list of CandidateSet
            Two lists of committees.

    Returns
    -------
        bool
    """
    for committee in committees1 + committees2:
        if not isinstance(committee, set):
            raise ValueError("Input has to be two lists of sets.")
    return all(committee in committees1 for committee in committees2) and all(
        committee in committees2 for committee in committees1
    )


def verify_expected_committees_equals_actual_committees(
    actual_committees, expected_committees, resolute=False, shortname="Rule"
):
    """
    Verify whether a voting rule returned the correct output. Raises exceptions if not.

    Check whether two lists of committees (`actual_committees` and `expected_committees`) are
    equivalent. Raise RuntimeError if not.

    Parameters
    ----------
        actual_committees : list of CandidateSet
            Output of an ABC voting rule.

        expected_committees : list of CandidateSet
            Expected output of this voting rule.

        resolute : bool, default=False
            If `True`, raise RuntimeError if more `actual_committees` does not have length 1.

        shortname : str, optional
            Name of rule used for Exception messages.

    Returns
    -------
        None
    """
    if resolute:
        if len(actual_committees) != 1:
            raise RuntimeError(
                f"Did not return exactly one committee (but {len(actual_committees)}) "
                f"for resolute=True."
            )
        if actual_committees[0] not in expected_committees:
            raise ValueError(
                f"{shortname} returns {actual_committees}, expected {expected_committees}"
            )
    else:
        if not compare_list_of_committees(actual_committees, expected_committees):
            raise ValueError(
                f"{shortname} returns {actual_committees}, expected {expected_committees}"
            )


def dominate(profile, committee1, committee2):
    """
    Test whether committee `committee1` dominates committee `committee2`.

    That is, test whether each voter in the profile has at least as many approved candidates
    in `committee1` as in `committee2`, and there is at least one voter with strictly more approved
    candidates in `committee1`.

    Parameters
    ----------
    profile : abcvoting.preferences.Profile
        A profile.
    committee1, committee2 : iterable of int
        Two committees.

    Returns
    -------
    bool
    """

    committee1 = CandidateSet(committee1)
    committee2 = CandidateSet(committee2)

    # iterate through all voters
    for voter in profile:
        # check if there are at least as many approved candidates in `committee1`
        # as in `committee2`
        if len(voter.approved & committee1) < len(voter.approved & committee2):
            return False

    # if not yet returned by now, then check for condition whether there is a voter with strictly
    # more preferred candidates in dominating committee than in input committee
    for voter in profile:
        # check if there are for some voter strictly more preferred candidates in `committee1`
        # than in `committee2`
        if len(voter.approved & committee1) > len(voter.approved & committee2):
            return True

    # If function has still not returned by now, then it means that `committee1` does not
    # dominate `committee2`.
    return False


def binom(n, k):
    """
    Compute a binomial coefficient (n choose k).

    Parameters
    ----------
        n, k : int
            Positive integers.

    Returns
    -------
        int
    """
    # could be replace by math.comb(n, k) in Python >= 3.8
    try:
        return math.factorial(n) // math.factorial(k) // math.factorial(n - k)
    except ValueError:
        return 0


def isclose(x, y):
    """
    Compare two floats using the abcvoting default values for absolute and relative tolerance.

    Parameters
    ----------
        x, y : float
            Two floats.

    Returns
    -------
        bool
    """
    return math.isclose(x, y, rel_tol=FLOAT_ISCLOSE_REL_TOL, abs_tol=FLOAT_ISCLOSE_ABS_TOL)


def time_it(func):
    def wrapper_function(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__}({kwargs}) needed {perf_counter() - start} seconds")
        return result

    return wrapper_function
