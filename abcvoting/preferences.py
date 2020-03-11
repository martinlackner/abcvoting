"""
Dichotomous (approval) preferences and preference profiles
Voters are indexed by 0, ..., len(profile)
Candidates are indexed by 0, ..., profile.num_cand
"""


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
        if names:
            if len(names) < num_cand:
                raise ValueError("names " + str(names) + " has length "
                                 + str(len(names)) + " < num_cand ("
                                 + str(num_cand) + ")")
            self.names = [str(names[i]) for i in range(num_cand)]

    def __len__(self):
        return len(self.preferences)

    def add_preferences(self, pref):
        if type(pref) is list:
            if len(pref) == 0:
                return
            if type(pref[0]) is int:
                # list of integers
                self.preferences.append(DichotomousPreferences(pref))
            else:
                # list of integer-lists or DichotomousPreferences
                for p in pref:
                    if type(p) is list:
                        newpref = DichotomousPreferences(p)
                        newpref.is_valid(self.num_cand)
                        self.preferences.append(newpref)
                    elif isinstance(p, DichotomousPreferences):
                        p.is_valid(self.num_cand)
                        self.preferences.append(p)
                    else:
                        raise TypeError("Object of type " + str(type(p)) +
                                        " not suitable as preferences")
        elif isinstance(pref, DichotomousPreferences):
            pref.is_valid(self.num_cand)
            self.preferences.append(pref)
        else:
            raise TypeError("Object of type " + str(type(pref)) +
                            " not suitable as preferences")

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
            return ("profile with %d votes and %d candidates:\n "
                    % (len(self.preferences), self.num_cand)
                    + ",\n ".join(map(str, self.preferences)))
        else:
            output = ("weighted profile with %d votes and %d candidates:\n"
                      % (len(self.preferences), self.num_cand))
            for p in self.preferences:
                output += " " + str(p.weight) + " * " + str(p) + ",\n"
            output = output[:-2]
            return output

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


class DichotomousPreferences():
    def __init__(self, approved, weight=1):
        self.approved = set(approved)
        if approved:  # empty approval sets are fine
            self.is_valid(max(approved) + 1)
        self.weight = weight

    def __str__(self):
        return str(list(self.approved))

    def __len__(self):
        return len(self.approved)

    def __iter__(self):
        return iter(self.approved)

    def is_valid(self, num_cand):
        for c in self.approved:
            if c < 0 or c >= num_cand:
                raise ValueError(str(self) + " not valid for num_cand = " +
                                 str(num_cand))
        return True
