# Basic functions for committees (i.e., subsets of candidates)

# Author: Martin Lackner


# sorts a list of committees, converts them to lists, and removes duplicates
def sort_committees(committees):
    return [sorted(list(c)) for c in sorted(set(map(tuple, committees)))]


# verifies whether a sufficient number of approved candidates exists
def enough_approved_candidates(profile, committeesize):
    appr = set()
    for pref in profile.preferences:
        appr.update(pref.approved)
    if len(appr) < committeesize:
        raise Exception("committeesize = " + str(committeesize)
                        + " is larger than number of approved candidates")


# nicely print a list of committees
def print_committees(committees, print_max=10):
    if committees is None:
        print "Error: no committees returned"
        return
    if len(committees) == 1:
        print " 1 committee"
    else:
        if len(committees) > print_max:
            print " ", len(committees), "committees,",
            print "printing ", print_max, "of them"
        else:
            print " ", len(committees), "committees"
    for com in sorted(map(tuple, committees[:print_max])):
        print "    {", ", ".join(map(str, com)), "}"
    print
