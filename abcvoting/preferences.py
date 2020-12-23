"""
Preference profiles and approval sets
 Voters are indexed by 0, ..., len(profile)
 Candidates are indexed by 0, ..., profile.num_cand
 Each voter is specified by an approval set.
 Approval sets are sets of candidates.
"""


from abcvoting.misc import str_candset
from collections import OrderedDict


class Profile(object):
    """
    Preference profiles consisting of approval sets.

    Properties
    ----------
    num_cand : int
        number of candidates or alternatives, denoted with m in the survey paper
    cand_names : iterable of str
        symbolic names for the candidates, defaults to '1', '2', ..., str(num_cand)
    preferences : list of DichotomousPreferences
        approved candidates for each voter, use `Profile.add_preferences()` to add  preferences

    """
    def __init__(self, num_cand, cand_names=None):
        if num_cand <= 0:
            raise ValueError(str(num_cand) +
                             " is not a valid number of candidates")
        self.num_cand = num_cand
        self.approval_sets = []  # entries correspond to voters
        self.cand_names = [str(c) for c in range(num_cand)]
        if cand_names:
            if len(cand_names) < num_cand:
                raise ValueError("cand_names " + str(cand_names) + " has length "
                                 + str(len(cand_names)) + " < num_cand ("
                                 + str(num_cand) + ")")
            self.cand_names = [str(cand_names[i]) for i in range(num_cand)]

    def __len__(self):
        return len(self.approval_sets)

    def add_voter(self, pref):
        """
        Adds a set of approved candidates of one voter to the preference profile.

        Parameters
        ----------
        pref : ApprovalSet or iterable of int

        """
        if isinstance(pref, ApprovalSet):
            appr_set = pref
        else:
            appr_set = ApprovalSet(pref)

        # this check is a bit redundant, but needed to check for consistency with self.num_cand
        appr_set.check_valid(self.num_cand)
        self.approval_sets.append(appr_set)

    def add_voters(self, prefs):
        """
        Adds several voters to the preference profile.
        Each voter is specified by a set (or list) of approved candidates
        or by an object of type ApprovalSet.

        Parameters
        ----------
        prefs : iterable of ApprovalSet or iterable of iterables of int

        """
        for p in prefs:
            self.add_voter(p)

    def totalweight(self):
        return sum(pref.weight for pref in self.approval_sets)

    def has_unit_weights(self):
        for p in self.approval_sets:
            if p.weight != 1:
                return False
        return True

    def __iter__(self):
        return iter(self.approval_sets)

    def __getitem__(self, i):
        return self.approval_sets[i]

    def __str__(self):
        if self.has_unit_weights():
            output = ("profile with %d votes and %d candidates:\n"
                      % (len(self.approval_sets), self.num_cand))
            for p in self.approval_sets:
                output += " " + str_candset(p.approved, self.cand_names) + ",\n"
        else:
            output = ("weighted profile with %d votes and %d candidates:\n"
                      % (len(self.approval_sets), self.num_cand))
            for p in self.approval_sets:
                output += (" " + str(p.weight) + " * "
                           + str_candset(p.approved, self.cand_names) + ",\n")
        return output[:-2]

    def party_list(self):
        """
        Is this party a party-list profile?
        In a party-list profile all approval sets are either
        disjoint or equal (see https://arxiv.org/abs/1704.02453).
        """
        for pref1 in self.approval_sets:
            for pref2 in self.approval_sets:
                if ((len(pref1.approved & pref2.approved)
                     not in [0, len(pref1.approved)])):
                    return False
        return True

    def str_compact(self):
        compact = OrderedDict()
        for p in self.approval_sets:
            if tuple(p.approved) in compact:
                compact[tuple(p.approved)] += p.weight
            else:
                compact[tuple(p.approved)] = p.weight
        if self.has_unit_weights():
            output = ""
        else:
            output = "weighted "
        output += ("profile with %d votes and %d candidates:\n"
                   % (len(self.approval_sets), self.num_cand))
        for apprset in compact:
            output += (" " + str(compact[apprset]) + " x "
                       + str_candset(apprset, self.cand_names) + ",\n")
        output = output[:-2]
        if not self.has_unit_weights():
            output += "\ntotal weight: " + str(self.totalweight())
        output += "\n"

        return output

    def aslist(self):
        return [list(pref.approved) for pref in self.approval_sets]


class ApprovalSet:
    """
    A set of approved candidates by one voter.
    """
    def __init__(self, approved, weight=1):
        self.approved = set(approved)
        self.weight = weight

        # does not check for num_cand, because not known here
        self.check_valid(approved_raw=approved)

    def __str__(self):
        return str(list(self.approved))

    def __len__(self):
        return len(self.approved)

    def __iter__(self):
        return iter(self.approved)

    def check_valid(self, num_cand=float('inf'), approved_raw=None):
        """
        Check if approved candidates are given as non-negative integers. If `num_cand` is known,
        also check if they are too large. Double entries are check if approved_raw is given as
        list or tuple (or similar).
        """
        if approved_raw is not None and len(self.approved) < len(approved_raw):
            raise ValueError(f"double entries found in list of approved candidates: {approved_raw}")

        # note: empty approval sets are fine
        for candidate in self.approved:
            if not isinstance(candidate, int):
                raise TypeError("Object of type " + str(type(candidate)) +
                                " not suitable as preferences")
            if candidate < 0 or candidate >= num_cand:
                raise ValueError(str(self) + " not valid for num_cand = " + str(num_cand))
