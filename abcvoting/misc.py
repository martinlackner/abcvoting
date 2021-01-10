"""
Miscellaneous functions for committees (i.e., subsets of candidates)
"""


from __future__ import print_function


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
    border = symbol[0] * len(text) + "\n"
    return border + text + "\n" + border
