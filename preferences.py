# Dichotomous (approval) preferences and profiles

# Author: Martin Lackner


class Profile(object):
    def __init__(self, num_cand, cand_names=None):
        self.num_cand = num_cand
        self.preferences = []
        self.cand_names = cand_names if cand_names else {}

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
                        self.preferences.append(DichotomousPreferences(p))
                    else:
                        p.is_valid(self.num_cand)
                        self.preferences.append(p)
        elif isinstance(pref, DichotomousPreferences):
            pref.is_valid(self.num_cand)
            self.preferences.append(pref)
        else:
            raise Exception("Object of type", type(pref), "not suitable as preferences")

    def totalweight(self):
        return reduce(lambda acc, prof: acc + prof.weight, self.preferences, 0)

    def has_unit_weights(self):
        for p in self.preferences:
            if p.weight != 1:
                return False
        return True

    def __iter__(self):
        return iter(self.preferences)

    def __str__(self):
        return 'profile with %d votes and %d candidates: ' % (len(self.preferences), self.num_cand) + ', '.join(map(str, self.preferences))


class DichotomousPreferences():
    def __init__(self, approved, weight=1):
        self.approved = set(approved)
        self.is_valid(max(approved) + 1)
        self.weight = weight

    def __str__(self):
        return str(list(self.approved))

    def is_valid(self, num_cand):
        for c in self.approved:
            if c < 0 or c >= num_cand:
                raise Exception(self, " not valid for num_cand =", num_cand)

        return True
