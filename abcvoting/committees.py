"""
Basic functions for committees (i.e., subsets of candidates)
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


# nicely print a list of committees
def print_committees(committees, print_max=10, names=None):
    if committees is None:
        print("Error: no committees returned")
        return
    if len(committees) == 1:
        print(" 1 committee:")
    else:
        if len(committees) > print_max:
            print(" ", len(committees), "committees:,", end=' ')
            print("printing ", print_max, "of them")
        else:
            print(" ", len(committees), "committees:")
    if names is None:
        for comm in sorted(map(tuple, committees[:print_max])):
            print("    {" + ", ".join(map(str, comm)) + "}")
    else:
        for comm in sorted(map(tuple, committees[:print_max])):
            namedcom = [names[c] for c in comm]
            print("    {" + ", ".join(map(str, namedcom)) + "}")
    print()


# Hamming distance
def hamming(a, b):
    diffs = ([x for x in a if x not in b] +
             [x for x in b if x not in a])
    return len(diffs)
