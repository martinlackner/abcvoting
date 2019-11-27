# Dichotomous (approval) preferences and profiles


class Profile(object):
    def __init__(self, num_cand):
        if num_cand <= 0:
            raise ValueError(str(num_cand) +
                             " is not a valid number of candidates")
        self.num_cand = num_cand
        self.preferences = []

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
                        raise Exception("Object of type " + str(type(p)) +
                                        " not suitable as preferences")
        elif isinstance(pref, DichotomousPreferences):
            pref.is_valid(self.num_cand)
            self.preferences.append(pref)
        else:
            raise Exception("Object of type " + str(type(p)) +
                            " not suitable as preferences")

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
        if self.has_unit_weights():
            return ("profile with %d votes and %d candidates: "
                    % (len(self.preferences), self.num_cand)
                    + ", ".join(map(str, self.preferences)))
        else:
            output = ("weighted profile with %d votes and %d candidates: "
                      % (len(self.preferences), self.num_cand))
            for p in self.preferences:
                output += str(p.weight) + "*" + str(p) + ", "
            return output


class DichotomousPreferences():
    def __init__(self, approved, weight=1):
        self.approved = set(approved)
        if approved:
            self.is_valid(max(approved) + 1)
        else:
            self.is_valid(0)
        self.weight = weight

    def __str__(self):
        return str(list(self.approved))

    def is_valid(self, num_cand):
        for c in self.approved:
            if c < 0 or c >= num_cand:
                raise Exception(str(self) + " not valid for num_cand = " +
                                str(num_cand))

        return True
