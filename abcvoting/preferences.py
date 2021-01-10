"""
Preference profiles and voters
 Preference profiles consist of voters.
 Voters in a profile are indexed by 0, ..., len(profile)-1
 Candidates are indexed by 0, ..., profile.num_cand-1
 The preferences of voters are specified by approval sets, which are sets of candidates.
"""


from abcvoting.misc import str_set_of_candidates
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
    _voters : list of Voter
        the list of voters, use `Profile.add_voter()` or `Profile.add_voters()`
        to add voters

    """

    def __init__(self, num_cand, cand_names=None):
        if num_cand <= 0:
            raise ValueError(str(num_cand) + " is not a valid number of candidates")
        self.candidates = list(range(num_cand))
        self._voters = []
        self.cand_names = [str(cand) for cand in range(num_cand)]
        if cand_names:
            if len(cand_names) < num_cand:
                raise ValueError(
                    f"cand_names {str(cand_names)} has length {len(cand_names)}"
                    f"< num_cand ({num_cand})"
                )
            self.cand_names = [str(cand_names[i]) for i in range(num_cand)]

    @property
    def num_cand(self):  # number of candidates
        return len(self.candidates)

    def __len__(self):
        return len(self._voters)

    def add_voter(self, voter):
        """
        Adds a set of approved candidates of one voter to the preference profile.

        Parameters
        ----------
        voter : Voter or iterable of int

        """
        # note that we trust that each set in self._voters is a unique object even if
        # voter.approved might not be unique, because it is used as dict key
        # (see e.g. the variable utility in abcrules_gurobi or propositionA3.py)
        if isinstance(voter, Voter):
            _voter = voter
        else:
            _voter = Voter(voter)

        # this check is a bit redundant, but needed to check for consistency with self.num_cand
        _voter.check_valid(self.num_cand)
        self._voters.append(_voter)

    def add_voters(self, voters):
        """
        Adds several voters to the preference profile.
        Each voter is specified by a set (or list) of approved candidates
        or by an object of type Voter.

        Parameters
        ----------
        voters : iterable of Voter or iterable of iterables of int

        """
        for voter in voters:
            self.add_voter(voter)

    def totalweight(self):
        return sum(voter.weight for voter in self._voters)

    def has_unit_weights(self):
        return all(voter.weight == 1 for voter in self._voters)

    def __iter__(self):
        return iter(self._voters)

    def __getitem__(self, i):
        return self._voters[i]

    def __str__(self):
        if self.has_unit_weights():
            output = f"profile with {len(self._voters)} votes and {self.num_cand} candidates:\n"
            for voter in self._voters:
                output += " " + str_set_of_candidates(voter.approved, self.cand_names) + ",\n"
        else:
            output = (
                f"weighted profile with {len(self._voters)} votes"
                f" and {self.num_cand} candidates:\n"
            )
            for voter in self._voters:
                output += f" {voter.weight} * "
                output += f"{str_set_of_candidates(voter.approved, self.cand_names)} ,\n"
        return output[:-2]

    def is_party_list(self):
        """
        Is this party a party-list profile?
        In a party-list profile all approval sets are either
        disjoint or equal (see https://arxiv.org/abs/1704.02453).
        """
        return all(
            len(voter1.approved & voter2.approved) in (0, len(voter1.approved))
            for voter1 in self._voters
            for voter2 in self._voters
        )

    def str_compact(self):
        compact = OrderedDict()
        for voter in self._voters:
            if tuple(voter.approved) in compact:
                compact[tuple(voter.approved)] += voter.weight
            else:
                compact[tuple(voter.approved)] = voter.weight
        if self.has_unit_weights():
            output = ""
        else:
            output = "weighted "
        output += "profile with %d votes and %d candidates:\n" % (len(self._voters), self.num_cand)
        for approval_set in compact:
            output += (
                " "
                + str(compact[approval_set])
                + " x "
                + str_set_of_candidates(approval_set, self.cand_names)
                + ",\n"
            )
        output = output[:-2]
        if not self.has_unit_weights():
            output += "\ntotal weight: " + str(self.totalweight())
        output += "\n"

        return output

    def aslist(self):
        return [list(voter.approved) for voter in self._voters]


class Voter:
    """
    A set of approved candidates by one voter.
    """

    def __init__(self, approved, weight=1):
        self.approved = set(approved)  # approval set, i.e., the set of approved candidates
        self.weight = weight

        # does not check for num_cand, because not known here
        self.check_valid(approved_raw=approved)

    def __str__(self):
        return str(list(self.approved))

    # some shortcuts, removed for clarity
    #
    # def __len__(self):
    #    return len(self.approved)
    #
    # def __iter__(self):
    #     return iter(self.approved)

    def check_valid(self, num_cand=float("inf"), approved_raw=None):
        """
        Check if approved candidates are given as non-negative integers. If `num_cand` is known,
        also check if they are too large. Double entries are check if approved_raw is given as
        list or tuple (or similar).
        """
        if approved_raw is not None and len(self.approved) < len(approved_raw):
            raise ValueError(
                f"double entries found in list of approved candidates: {approved_raw}"
            )

        # note: empty approval sets are fine
        for candidate in self.approved:
            if not isinstance(candidate, int):
                raise TypeError(
                    f"Object of type {str(type(candidate))} not suitable as candidate, "
                    f"only positive integers allowed."
                )
            if candidate < 0 or candidate >= num_cand:
                raise ValueError(str(self) + " not valid for num_cand = " + str(num_cand))
