"""
Miscellaneous functions for committees (i.e., subsets of candidates)
"""


from __future__ import print_function


def sort_committees(committees):
    """
    sorts a list of committees,
    converts them to lists, and removes duplicates
    """
    return [sorted(list(c)) for c in sorted(set(map(tuple, committees)))]


def check_enough_approved_candidates(profile, committeesize):
    """
    verifies whether a sufficient number of approved candidates exists
    """
    appr = set()
    for pref in profile:
        appr.update(pref)
    if len(appr) < committeesize:
        raise ValueError("committeesize = " + str(committeesize)
                         + " is larger than number of approved candidates")


def str_candset(candset, cand_names=None):
    """
    nicely format a single committee
    """
    if cand_names is None:
        namedset = [str(c) for c in candset]
    else:
        namedset = [cand_names[c] for c in candset]
    return "{" + ", ".join(map(str, namedset)) + "}"


def str_candsets(committees, cand_names=None):
    """
    nicely format a list of committees
    """
    output = ""
    for comm in sorted(map(tuple, committees)):
        output += " " + str_candset(comm, cand_names) + "\n"
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


def hamming(a, b):
    """Hamming distance between sets `a` and `b`.

    The Hamming distance for sets is the size of their symmetric difference,
    or, equivalently, the usual Hamming distance when sets are viewed as 0-1-strings.

    Parameters
    ----------
    a, b : iterable of int
        The two sets, for which the Hamming distance is computed."""
    diffs = ([x for x in a if x not in b] +
             [x for x in b if x not in a])
    return len(diffs)


def header(text, symbol="-"):
    border = symbol[0] * len(text) + "\n"
    return border + text + "\n" + border
