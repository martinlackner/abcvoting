"""
Preference profiles and voters.

.. important::

    - Preference profiles consist of voters.
    - Voters in a profile are indexed by `0`, ..., `len(profile)-1`
    - Candidates are indexed by `0`, ..., `profile.num_cand-1`
    - The preferences of voters are specified by approval sets, which are sets of candidates.

"""

from collections import OrderedDict
from abcvoting import misc


class Profile:
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

        if cand_names:
            if len(cand_names) < num_cand:
                raise ValueError(
                    f"cand_names {str(cand_names)} has length {len(cand_names)}"
                    f"< num_cand ({num_cand})"
                )
            self.cand_names = [str(cand_names[i]) for i in range(num_cand)]

    @property
    def num_cand(self):  # number of candidates
        """Number of candidates."""
        return len(self.candidates)

    def approved_candidates(self):
        """
        A set of all candidates approved by at least one voter.

        Returns
        -------
        set of int
        """
        _approved_candidates = set()
        for voter in self._voters:
            _approved_candidates.update(voter.approved)
        return _approved_candidates

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

    def total_weight(self):
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

    def convert_to_unit_weights(self):
        """
        Convert all voters with weights into the appropropriate number of unit-weight copies.

        Only works if weights are integers.

        Returns
        -------
            None

        Examples
        --------
        .. doctest::

            >>> profile = Profile(num_cand=3)
            >>> profile.add_voter(Voter([0, 1], weight=2))
            >>> profile.add_voter(Voter([2], weight=1))
            >>> print(profile)
            weighted profile with 2 voters and 3 candidates:
             voter 0:   2 * {0, 1},
             voter 1:   1 * {2}
            >>> profile.convert_to_unit_weights()
            >>> print(profile)
            profile with 3 voters and 3 candidates:
             voter 0:   {0, 1},
             voter 1:   {0, 1},
             voter 2:   {2}
        """
        new_voters = []
        for voter in self._voters:
            try:
                for _ in range(voter.weight):
                    new_voters.append(Voter(voter.approved))
            except TypeError:
                raise TypeError(
                    "Converting a profile to unit weights is only possible with integer weights."
                )
        self._voters = new_voters

    def convert_to_weighted(self):
        """
        Merge all voters with the same approval set into a single voter with appropropriate weight.

        Returns
        -------
            None

        Examples
        --------
        .. doctest::

            >>> profile = Profile(num_cand=3)
            >>> profile.add_voters([[0, 1], [0, 1], [2]])
            >>> print(profile)
            profile with 3 voters and 3 candidates:
             voter 0:   {0, 1},
             voter 1:   {0, 1},
             voter 2:   {2}
            >>> profile.convert_to_weighted()
            >>> print(profile)
            weighted profile with 2 voters and 3 candidates:
             voter 0:   2 * {0, 1},
             voter 1:   1 * {2}
        """
        approval_sets = {tuple(sorted(voter.approved)) for voter in self._voters}
        weights = {appr: 0 for appr in approval_sets}
        for voter in self._voters:
            weights[tuple(sorted(voter.approved))] += voter.weight
        self._voters = [Voter(appr, weight=weight) for appr, weight in weights.items()]

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
            output = f"profile with {len(self._voters)} voters and {self.num_cand} candidates:\n"
        else:
            output = (
                f"weighted profile with {len(self._voters)} voters"
                f" and {self.num_cand} candidates:\n"
            )
        for vi, voter in enumerate(self._voters):
            output += f" voter {str(vi) + ':':4s} "
            if not self.has_unit_weights():
                output += f"{voter.weight} * "
            output += f"{misc.str_set_of_candidates(voter.approved, self.cand_names)},\n"
        return output[:-2]

    def copy(self):
        """
        Return a copy of the profile.

        This is a deep copy, i.e., all Voter objects are copied too.

        Returns
        -------
            Profile
        """
        copy_profile = Profile(num_cand=self.num_cand)
        copy_profile.add_voters(self._voters)
        return copy_profile

    __copy__ = copy
    __deepcopy__ = copy

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
        output += f"profile with {len(self._voters)} voters and {self.num_cand} candidates:\n"
        for approval_set in compact:
            output += (
                f" {compact[approval_set]} x "
                f"{misc.CandidateSet(approval_set).str_with_names(self.cand_names)},\n"
            )
        output = output[:-2]
        if not self.has_unit_weights():
            output += "\ntotal weight: " + str(self.total_weight())
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

            If `weight` is an integer, the voter is interpreted as if there are `weight`
            many voters with the same approval set.

        num_cand : int, optional
            The maximum number of candidates. Used only for checks.

            If this `num_cand` is provided, it is verified that `approved` does not contain
            numbers `>= num_cand`.
    """

    def __init__(self, approved, weight=1, num_cand=None):
        self.approved = misc.CandidateSet(approved, num_cand=num_cand)
        self.weight = weight

        # check weights
        if self.weight <= 0:
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
