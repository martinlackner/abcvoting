"""
Dichotomous (approval) preferences and preference profiles
Voters are indexed by 0, ..., len(profile)
Candidates are indexed by 0, ..., profile.num_cand
"""


from abcvoting.misc import str_candset
from collections import OrderedDict


class Profile(object):
    """
    Preference profiles
    """
    def __init__(self, num_cand, names=None):
        if num_cand <= 0:
            raise ValueError(str(num_cand) +
                             " is not a valid number of candidates")
        self.num_cand = num_cand
        self.preferences = []
        self.names = [str(c) for c in range(num_cand)]
        if names:
            if len(names) < num_cand:
                raise ValueError("names " + str(names) + " has length "
                                 + str(len(names)) + " < num_cand ("
                                 + str(num_cand) + ")")
            self.names = [str(names[i]) for i in range(num_cand)]

    def __len__(self):
        return len(self.preferences)

    def add_preference(self, preference):
        """Adds a set of approved candidates of one voter to the preference profile.

        Parameters
        ----------
        preference : DichotomousPreferences or iterable of int

        """
        if isinstance(preference, DichotomousPreferences):
            dichotomous_pref = preference
        else:
            dichotomous_pref = DichotomousPreferences(preference)

        # this check is a bit redundant, but needed to check for consistency with self.num_cand
        dichotomous_pref.check_valid(self.num_cand)
        self.preferences.append(dichotomous_pref)

    def add_preferences(self, preferences):
        """Add sets of approved candidates for many voters to the preference profile.

        Parameters
        ----------
        preferences : iterable of DichotomousPreferences or iterable of iterables of int

        """
        for p in preferences:
            self.add_preference(p)

    def totalweight(self):
        return sum(pref.weight for pref in self.preferences)

    def has_unit_weights(self):
        for p in self.preferences:
            if p.weight != 1:
                return False
        return True

    def __iter__(self):
        return iter(self.preferences)

    def __getitem__(self, i):
        return self.preferences[i]

    def __str__(self):
        if self.has_unit_weights():
            output = ("profile with %d votes and %d candidates:\n"
                      % (len(self.preferences), self.num_cand))
            for p in self.preferences:
                output += " " + str_candset(p.approved, self.names) + ",\n"
        else:
            output = ("weighted profile with %d votes and %d candidates:\n"
                      % (len(self.preferences), self.num_cand))
            for p in self.preferences:
                output += (" " + str(p.weight) + " * "
                           + str_candset(p.approved, self.names) + ",\n")
        return output[:-2]

    def party_list(self):
        """
        Is this party a party-list profile?
        In a party-list profile all approval sets are either
        disjoint or equal (see https://arxiv.org/abs/1704.02453).
        """
        for pref1 in self.preferences:
            for pref2 in self.preferences:
                if ((len(pref1.approved & pref2.approved)
                     not in [0, len(pref1.approved)])):
                    return False
        return True

    def str_compact(self):
        compact = OrderedDict()
        for p in self.preferences:
            if tuple(p.approved) in compact:
                compact[tuple(p.approved)] += p.weight
            else:
                compact[tuple(p.approved)] = p.weight
        if self.has_unit_weights():
            output = ""
        else:
            output = "weighted "
        output += ("profile with %d votes and %d candidates:\n"
                   % (len(self.preferences), self.num_cand))
        for apprset in compact:
            output += (" " + str(compact[apprset]) + " x "
                       + str_candset(apprset, self.names) + ",\n")
        output = output[:-2]
        if not self.has_unit_weights():
            output += "\ntotal weight: " + str(self.totalweight())
        output += "\n"

        return output

    def aslist(self):
        return [list(pref.approved) for pref in self.preferences]


class DichotomousPreferences:
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
        """Check if approved candidates are given as non-negative integers. If `num_cand` is known,
        also check if they are too large. Double entries are check if approved_raw is given as
        list or tuple (or similar)."""
        if approved_raw is not None and len(self.approved) < len(approved_raw):
            raise ValueError(f"double entries found in list of approved candidates: {approved_raw}")

        # note: empty approval sets are fine
        for candidate in self.approved:
            if not isinstance(candidate, int):
                raise TypeError("Object of type " + str(type(candidate)) +
                                " not suitable as preferences")
            if candidate < 0 or candidate >= num_cand:
                raise ValueError(str(self) + " not valid for num_cand = " + str(num_cand))
