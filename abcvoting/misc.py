"""
Miscellaneous functions for committees (i.e., subsets of candidates)
"""


from __future__ import print_function


# sorts a list of committees,
# converts them to lists, and removes duplicates
def sort_committees(committees):
    return [sorted(list(c)) for c in sorted(set(map(tuple, committees)))]


# verifies whether a sufficient number of approved candidates exists
def enough_approved_candidates(profile, committeesize):
    appr = set()
    for pref in profile:
        appr.update(pref)
    if len(appr) < committeesize:
        raise ValueError("committeesize = " + str(committeesize)
                         + " is larger than number of approved candidates")


def str_candset(candset, names=None):
    if names is None:
        namedset = [str(c) for c in candset]
    else:
        namedset = [names[c] for c in candset]
    return "{" + ", ".join(map(str, namedset)) + "}"


def str_candsets(committees, names=None):
    """
    nicely format a list of committees
    """
    output = ""
    for comm in sorted(map(tuple, committees)):
        output += " " + str_candset(comm, names) + "\n"
    return output


def str_committees_header(committees, winning=False):
    """
    nicely format a heaer for a list of committees,
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
    """Hamming distance"""
    diffs = ([x for x in a if x not in b] +
             [x for x in b if x not in a])
    return len(diffs)


def header(text, symbol="-"):
    border = symbol[0] * len(text) + "\n"
    return border + text + "\n" + border
