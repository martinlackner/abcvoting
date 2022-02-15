"""
Preference profiles and voters

.. important::

    - Preference profiles consist of voters.
    - Voters in a profile are indexed by `0`, ..., `len(profile)-1`
    - Candidates are indexed by `0`, ..., `profile.num_cand-1`
    - The preferences of voters are specified by approval sets, which are sets of candidates.

"""


from collections import OrderedDict
from abcvoting import misc


class Profile(object):
    """
    Approval profiles.

    Approval profiles are a list of voters, each of which has preferences
    expressed as approval sets.

    Parameters
    ----------

        num_cand : int
            Number of candidates in this profile.

            Remark: Not every candidate has to be approved by a voter.

        cand_names : list of str or str, optional
            List of symbolic names for every candidate.

            Defaults to `[1, 2, ..., str(num_cand)]`.

            For example, for `num_cand=5` one could have `cand_names="abcde"`.

    Attributes
    ----------

        candidates : list of int

            List of all candidates, i.e., the list containing `0`, ..., `profile.num_cand-1`.

        cand_names : list of str or str

            Symbolic names for every candidate.

            Defaults to `["0", "1", ..., str(num_cand-1)]`.
    """

    def __init__(self, num_cand, cand_names=None):
        if num_cand <= 0:
            raise ValueError(str(num_cand) + " is not a valid number of candidates")
        self.candidates = list(range(num_cand))
        self.cand_names = [str(cand) for cand in range(num_cand)]

        self._voters = []  # Internal list of voters.
        # Use `Profile.add_voter()` or `Profile.add_voters()` to add voters
        self._approved_candidates = None  # internal. Use approved_candidates.

        if cand_names:
            if len(cand_names) < num_cand:
                raise ValueError(
                    f"cand_names {str(cand_names)} has length {len(cand_names)}"
                    f"< num_cand ({num_cand})"
                )
            self.cand_names = [str(cand_names[i]) for i in range(num_cand)]

    @property
    def num_cand(self):  # number of candidates
        """
        Number of candidates.
        """
        return len(self.candidates)

    @property
    def approved_candidates(self):
        """
        A list of all candidates approved by at least one voter.
        """
        if self._approved_candidates is None:
            self._approved_candidates = set()
            for voter in self._voters:
                self._approved_candidates.update(voter.approved)
        return self._approved_candidates

    def __len__(self):
        return len(self._voters)

    def _unique_voter(self, voter):
        # we ensure that each set in self._voters is a unique object even if
        # voter.approved might not be unique, because it is used as dict key
        # (see e.g. the variable utility in abcrules_gurobi or propositionA3.py)
        if isinstance(voter, Voter):
            _voter = Voter(voter.approved, weight=voter.weight, num_cand=self.num_cand)
        else:
            _voter = Voter(voter, num_cand=self.num_cand)

        # there might be new approved candidates,
        # but update self._approved_candidates only on demand
        self._approved_candidates = None

        return _voter

    def add_voter(self, voter):
        """
        Add a set of approved candidates of one voter to the preference profile.

        Parameters
        ----------
            voter : Voter or iterable of int
                The voter to be added.

        Returns
        -------
            None
        """

        # ensure that new voter is unique
        self._voters.append(self._unique_voter(voter))

    def add_voters(self, voters):
        """
        Add several voters to the preference profile.

        Each voter is specified by a set (or list) of approved candidates
        or by an object of type Voter.

        Parameters
        ----------
            voters : iterable of Voter or iterable of iterables of int
                The voters to be added.

        Returns
        -------
            None
        """
        for voter in voters:
            self.add_voter(voter)

    def totalweight(self):
        """
        Return the totol weight of all voters, i.e., the sum of weights.

        Returns
        -------
            int or Fraction
                Total weight.
        """
        return sum(voter.weight for voter in self._voters)

    def has_unit_weights(self):
        """
        Verify whether all voters in the profile have a weight of 1.

        Returns
        -------
            bool
        """
        return all(voter.weight == 1 for voter in self._voters)

    def __iter__(self):
        return iter(self._voters)

    def __getitem__(self, i):
        return self._voters[i]

    def __setitem__(self, i, voter):
        """
        Modify a voter in the preference profile.

        Parameters
        ----------
            voter : Voter or iterable of int
        """

        # ensure that new voter is unique
        self._voters[i] = self._unique_voter(voter)

    def __str__(self):
        if self.has_unit_weights():
            output = f"profile with {len(self._voters)} votes and {self.num_cand} candidates:\n"
            for voter in self._voters:
                output += " " + voter.str_with_names(self.cand_names) + ",\n"
        else:
            output = (
                f"weighted profile with {len(self._voters)} votes"
                f" and {self.num_cand} candidates:\n"
            )
            for voter in self._voters:
                output += f" {voter.weight} * "
                output += f"{misc.str_set_of_candidates(voter.approved, self.cand_names)} ,\n"
        return output[:-2]

    def is_party_list(self):
        """
        Check whether this profile is a party-list profile.

        In a party-list profile all approval sets are either disjoint or equal.

        Returns
        -------
            bool
        """
        return all(
            len(voter1.approved & voter2.approved) in (0, len(voter1.approved))
            for voter1 in self._voters
            for voter2 in self._voters
        )

    def str_compact(self):
        """
        Return a string that compactly summarizes the profile and its voters.

        Returns
        -------
            str
        """
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
                f" {compact[approval_set]} x "
                f"{misc.CandidateSet(approval_set).str_with_names(self.cand_names)},\n"
            )
        output = output[:-2]
        if not self.has_unit_weights():
            output += "\ntotal weight: " + str(self.totalweight())
        output += "\n"

        return output


class Voter:
    """
    A set of approved candidates by one voter.

    Parameters
    ----------
        approved : CandidateSet
            The set of approved candidates.

        weight : int or Fraction, default=1
            The weight of the voter.

            This should not be used as the number of voters with these approved candidates.

        num_cand : int, optional
            The maximum number of candidates. Used only for checks.

            If this `num_cand` is provided, it is verified that `approved` does not contain
            numbers `>= num_cand`.
    """

    def __init__(self, approved, weight=1, num_cand=None):
        self.approved = misc.CandidateSet(approved, num_cand=num_cand)
        self.weight = weight

        # check weights
        if not self.weight > 0:
            raise ValueError("Weight should be a number > 0.")

    def __str__(self):
        return str(self.approved)

    def str_with_names(self, cand_names=None):
        """
        Format a Voter, using the names of candidates (instead of indices) if provided.

        Parameters
        ----------

            cand_names : list of str or str, optional
                List of symbolic names for every candidate.

        Returns
        -------
            str
        """
        return self.approved.str_with_names(cand_names)

    # some shortcuts, removed for clarity
    #
    # def __len__(self):
    #    return len(self.approved)
    #
    # def __iter__(self):
    #     return iter(self.approved)
