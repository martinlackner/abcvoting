"""
Read preflib files (soi, toi, soc or toc)
"""


from __future__ import print_function
import os
from abcvoting.preferences import Profile, ApprovalSet
from math import ceil


class PreflibException(Exception):
    pass


def load_preflib_files_from_dir(dir_name, setsize=1, appr_percent=None):
    """Loads all election files (soi, toi, soc or toc) from the given dir.

    Parameters:
        dir_name: str
            path of directory to be searched for preflib files.

        setsize: int
            Number of top-ranked candidates that voters approve.
            In case of ties, more than setsize candidates are approved.

            setsize is ignored if appr_percent is used.

        appr_percent: float in (0, 1]
            Indicates which percentage of candidates of the ranking
            are approved (rounded up). In case of ties, more
            candidates are approved.
            E.g., if a voter has 10 candidates and this value is 0.75,
            then the approval set contains the top 8 candidates.

        Returns:
            profiles: list
                a list of profiles (type Profile)
    """
    file_dir, files = get_file_names(dir_name)
    files = [file_dir + f for f in files]

    profiles = []
    if file_dir is not None:
        files = sorted(files)
        for f in files:
            if (
                f.endswith(".soi")
                or f.endswith(".toi")
                or f.endswith(".soc")
                or f.endswith(".toc")
            ):
                profile = read_preflib_file(
                    f, setsize=setsize, appr_percent=appr_percent
                )
                profiles.append(profile)

    return profiles


def get_file_names(dir_name):
    files = []
    for (dir_path, _, filenames) in os.walk(dir_name):
        file_dir = dir_path
        files = filenames
        break  # do not consider subdirs
    if len(files) == 0:
        raise PreflibException("No files found in", dir_name)
    return file_dir, files


def get_appr_set(num_appr, ranking, candidate_map):
    # if num_appr = 1 and the ranking starts with empty set, interpret as empty ballot and
    # return set()
    if (
        num_appr == 1
        and ranking[0].strip()[0] == "{"
        and ranking[0].strip()[-1] == "}"
        and ranking[0].strip().replace("}", "").replace("{", "").strip() == ""
    ):
        return set()

    appr_set = set()
    tied = False
    for i in range(len(ranking)):
        rank = ranking[i].strip()
        if rank.startswith("{"):
            if not tied:
                tied = True
                rank = rank[1:]
            else:
                raise PreflibException(
                    "Invalid format for tied candidates: " + str(ranking)
                )
        if rank.endswith("}"):
            if tied:
                tied = False
                rank = rank[:-1]
            else:
                raise PreflibException(
                    "Invalid format for tied candidates: " + str(ranking)
                )
        rank = rank.strip()
        if len(rank) > 0:
            try:
                c = int(rank)
            except ValueError:
                raise PreflibException(
                    "Expected candidate number but encountered " + str(c) + ""
                )
            appr_set.add(c)
        if len(appr_set) >= num_appr and not tied:
            break
    if tied:
        raise PreflibException("Invalid format for tied candidates: " + str(ranking))
    if len(appr_set) < num_appr:
        # all candidates approved
        appr_set = set(candidate_map.keys())
    return appr_set


def read_preflib_file(filename, setsize=1, appr_percent=None, use_weights=False):
    """Reads a single preflib file (soi, toi, soc or toc).

    Parameters:

        filename: str
            Name of the preflib file.

        setsize: int
            Number of top-ranked candidates that voters approve.
            In case of ties, more than setsize candidates are approved.

            setsize is ignored if appr_percent is used.

        appr_percent: float in (0, 1]
            Indicates which percentage of candidates of the ranking
            are approved (rounded up). In case of ties, more
            candidates are approved.
            E.g., if a voter has 10 approved candidates and appr_percent is 0.75,
            then the approval set contains the top 8 candidates.

        use_weights: bool
            If False, treat vote count in preflib file as the number of duplicate ballots,
            i.e., the number of voters that have this approval set.
            If True, treat vote count as weight and use this weight in ApprovalSet.

    Returns:
        profile: Profile
            Preference profile extracted from preflib file,
            including names of candidates
    """
    if setsize <= 0:
        raise ValueError("Parameter setsize must be > 0")
    if appr_percent and (appr_percent <= 0.0 or appr_percent > 1.0):
        raise ValueError("Parameter appr_percent not in interval (0, 1]")
    with open(filename, "r") as f:
        line = f.readline()
        num_cand = int(line.strip())
        candidate_map = {}
        for _ in range(num_cand):
            parts = f.readline().strip().split(",")
            candidate_map[int(parts[0].strip())] = ",".join(parts[1:]).strip()

        parts = f.readline().split(",")
        try:
            voter_count, _, unique_orders = [int(p.strip()) for p in parts]
        except ValueError:
            raise PreflibException(
                f"Number of voters ill specified ({str(parts)}), should be triple of integers"
            )

        appr_sets = []
        lines = [line.strip() for line in f.readlines() if line.strip()]
        if len(lines) != unique_orders:
            raise PreflibException(
                f"Expected {unique_orders} lines that specify voters in the input, "
                f"encountered {len(lines)}"
            )

    for line in lines:
        parts = line.split(",")
        if len(parts) < 1:
            continue
        try:
            count = int(parts[0])
        except ValueError:
            raise PreflibException(
                f"Each ranking must start with count/weight ({line})"
            )
        ranking = parts[1:]  # ranking starts after count
        if len(ranking) == 0:
            raise PreflibException("Empty ranking: " + str(line))
        if appr_percent:
            num_appr = int(ceil(len(ranking) * appr_percent))
        else:
            num_appr = setsize
        appr_set = get_appr_set(num_appr, ranking, candidate_map)
        appr_sets.append((count, appr_set))

    # normalize candidates to 0, 1, 2, ...
    cand_names = []
    normalize_map = {}
    for c in candidate_map.keys():
        cand_names.append(candidate_map[c])
        normalize_map[c] = len(cand_names) - 1

    profile = Profile(num_cand, cand_names=cand_names)

    for count, appr_set in appr_sets:
        norm_appr_set = []
        for c in appr_set:
            norm_appr_set.append(normalize_map[c])
        if use_weights:
            profile.add_voter(ApprovalSet(norm_appr_set, weight=count))
        else:
            profile.add_voters([norm_appr_set] * count)
    if use_weights:
        if len(profile) != unique_orders:
            raise PreflibException(
                "Number of voters wrongly specified in preflib file."
            )
    else:
        if len(profile) != voter_count:
            raise PreflibException(
                "Number of voters wrongly specified in preflib file."
            )
    return profile


def write_profile_to_preflib_toi_file(profile, filename):
    """Writes profile in a preflib toi file

    Parameters:

        profile: Profile
            Profile to be written in preflib file.

        filename: str
            File name of the preflib file to be written.

    Returns:
        None
    """
    with open(filename, "w") as f:
        # write: number of candidates
        f.write(str(profile.num_cand) + "\n")
        # write: names of candidates
        for i in range(profile.num_cand):
            f.write(f"{i + 1}, {profile.cand_names[i]}\n")
        # write: info about number of voters and total weight
        total_weight = sum(appr_set.weight for appr_set in profile)
        f.write(f"{total_weight}, {total_weight}, {len(profile)}\n")
        # write: approval sets and weights
        for appr_set in profile:
            appr_set_string = ", ".join(str(cand + 1) for cand in appr_set)
            f.write(f"{appr_set.weight}, {{{appr_set_string}}}\n")
