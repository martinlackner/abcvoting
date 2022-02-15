"""
Miscellaneous functions for committees (i.e., subsets of candidates).
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

        super(CandidateSet, self).__init__(candidates)
        if len(candidates) != len(self):
            raise ValueError(f"CandidateSet initialized with duplicate elements ({candidates}).")

        for cand in candidates:
            if not isinstance(cand, int):
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
    return sorted([CandidateSet(committee) for committee in committees], key=str)


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
        named = sorted([str(cand_names[cand]) for cand in candset])
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
        else:
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
