"""
Miscellaneous functions for committees (i.e., subsets of candidates)
"""


def sorted_committees(committees):
    """
    sorts a list of committees, ensures that committees are sets
    """
    return sorted([set(committee) for committee in committees], key=str)


def check_enough_approved_candidates(profile, committeesize):
    """
    verifies whether a sufficient number of approved candidates exists
    """
    approved_candidates = set()
    for voter in profile:
        approved_candidates.update(voter.approved)
    if len(approved_candidates) < committeesize:
        raise ValueError(
            f"committeesize = {committeesize} is larger than"
            f"number of approved candidates ({len(approved_candidates)})"
        )


def str_set_of_candidates(candset, cand_names=None):
    """
    nicely format a single committee
    """
    if cand_names is None:
        namedset = [str(cand) for cand in candset]
    else:
        namedset = [cand_names[cand] for cand in candset]
    return "{" + ", ".join(map(str, namedset)) + "}"


def str_sets_of_candidates(committees, cand_names=None):
    """
    nicely format a list of committees
    """
    output = ""
    for committee in sorted(map(tuple, committees)):
        output += f" {str_set_of_candidates(committee, cand_names)}\n"
    return output


def str_committees_header(committees, winning=False):
    """
    nicely format a header for a list of committees,
    stating how many committees there are

    winning: write "winning committee" instead of "committee"
    """
    output = ""
    if committees is None or len(committees) < 1:
        if winning:
            return "No winning committees (this should not happen)"
        else:
            return "No committees"
    if winning:
        commstring = "winning committee"
    else:
        commstring = "committee"
    if len(committees) == 1:
        output += "1 " + commstring + ":"
    else:
        output += str(len(committees)) + " " + commstring + "s:"
    return output


def hamming(set1, set2):
    """Hamming distance between sets `a` and `b`.

    The Hamming distance for sets is the size of their symmetric difference,
    or, equivalently, the usual Hamming distance when sets are viewed as 0-1-strings.

    Parameters
    ----------
    set1, set2 : iterable of int
        The two sets, for which the Hamming distance is computed."""
    diffs = [x for x in set1 if x not in set2] + [x for x in set2 if x not in set1]
    return len(diffs)


def header(text, symbol="-"):
    """Returns a header string for `text`."""
    border = symbol[0] * len(text) + "\n"
    return border + text + "\n" + border


def compare_list_of_committees(list1, list2):
    """Check whether two lists of committees are equal when the order (and multiplicities)
    in these lists are ignored.
    To be precise, two lists are equal if every committee in list1 is contained in list2 and
    vice versa.
    Committees are, as usual, sets of positive integers.

    Parameters
    ----------
    list1, list2 : iterable of sets"""
    for committee in list1 + list2:
        assert isinstance(committee, set)
    return all(committee in list1 for committee in list2) and all(
        committee in list2 for committee in list1
    )


def verify_expected_committees_equals_actual_committees(
    actual_committees, expected_committees, resolute=False, shortname="Rule"
):
    """Check that two lists of committees are equivalent.

    Raise RuntimeError if not."""
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
