# load preflib files (SOI or TOI)


from __future__ import print_function
import os
from copy import copy

script_dir = os.path.dirname(__file__)


def load_election_files_from_dir(dir_name, max_approval_percent=1.0):
    """Loads all election files (toi, soi) from the given dir.

    Parameters:
        dir_name: str
            relative path to the root of this project.
        max_approval_percent: float (0, 1]
            how many of the candidates within a ranking  should be
            used for the approval set.
            E.g. if a ranking has 10 candidates, max_approval_percent is
            0.5 then 5 candidates are considered.
            In case of toi files this can be more if the last candidate
            that would normally be taken is tied with other candidates.
            E.g. 1, 2, 3, 4, {5, 6}, 7,8, 9, 10: with 0.5 it would end
            with 5 but 6 is also included. apprset=[1, 2, 3, 4, 5, 6]

        Returns:
            candidate_map: dict
                dict from normalized candidate ids to candidate names
            profile: list of lists
                list of approval set lists for each voter one
            used_candidate_count: int
                number of candidates that are within the profile
        """
    file_dir, files = get_file_names(dir_name)

    profiles = []
    if file_dir is not None:
        files = sorted(files)  # sorts from oldest to newest if name is sortable by date (YYYYMMDD)
        for f in files:
            if f.endswith('.soi') or f.endswith('.toi'):
                ''' # can be added if not all soi from a directory are needed.
                if from_date is not None or to_date is not None:
                    date = f.split("_")[-1].split(".")[-1]
                    if from_date is not None and date < from_date:
                        continue
                    if to_date is not None and date > to_date:
                        break
                '''
                candidate_map, profile, used_candidate_count = read_election_file(
                    f, max_approval_percent, file_dir=file_dir)
                profiles.append((candidate_map, profile, used_candidate_count))

    return profiles


def get_file_names(dir_name):
    input_path = os.path.join(script_dir, dir_name)
    files = []
    file_dir = None
    for (dir_path, _, filenames) in os.walk(input_path):
        file_dir = dir_path
        files = filenames
        break
    if len(files) == 0:
        raise Exception("No files found in", input_path)
    return file_dir, files


def get_vote(threshold, ranking):
    appr_set = []
    tied = False
    curr_voters = ""
    for i in range(len(ranking)):
        if threshold > 0:
            voter = ranking[i].strip()
            if not tied:
                if voter.startswith("{"):
                    tied = True
                    # count = 1
                    curr_voters = voter
                    if "}" in voter:
                        raise Exception("Single voter in {} is invalid")
                else:
                    add_candidate(voter, appr_set)
                    threshold -= 1
            else:
                curr_voters += "," + voter
                # count += 1
                if "}" in voter:
                    count = add_candidate(curr_voters, appr_set)
                    curr_voters = ""
                    threshold -= count  # or -= 1
                    # count = 0
                else:
                    continue
        else:
            break
    return appr_set


def add_candidate(rank, appr_set):
    candidate = rank.strip()
    if candidate.find("{") != -1:
        if candidate[0] != "{" or candidate[-1] != "}":
            raise Exception("Invalid format for tied candidates.",
                            rank)
        candidates = candidate[1:-1].split(",")
        for c in candidates:
            appr_set.append(int(c.strip()))
        return len(candidate)
    else:
        appr_set.append(int(candidate))
        return 1


def read_election_file(filename, max_approval_percent=0.8,
                       file_dir=None):
    """Reads a single election file (soi or toi).

    Parameters:

        filename: str
            Name of the file.

        max_approval_percent: float
            Indicates how many candidates of the ranking are approved.
            If a voter has 10 candidates and this value is 0.8,
            then 8 candidates are taken as approval set.

        file_dir: str
            Absolute path to the directory that contains the file.
            If None a relative path to this file is chosen.

    Returns:
        used_candidate_map: dict
            dictionary with normalized candidate ids to names.

        normalized_profile:  list
            list of lists that represent approval sets.

        used_candidate_count: int
            Number of candidates within the profile.

        """
    if file_dir is None:
        filename = os.path.join(script_dir, filename)
    else:
        filename = os.path.join(file_dir, filename)
    with open(filename, "r") as f:
        line = f.readline()
        candidate_count = int(line.strip())
        candidate_map = {}
        for i in range(candidate_count):
            parts = f.readline().strip().split(",")
            candidate_map[int(parts[0].strip())] = \
                ",".join(parts[1:]).strip()

        parts = f.readline().split(",")
        voter_count = int(parts[0].strip())
        # voter_sum = int(parts[1].strip())
        unique_orders = int(parts[2].strip())
        unique_candidates = set()
        profile = []
        for i in range(unique_orders):
            line = f.readline().strip()
            parts = line.split(",")
            count = int(parts[0])
            ranking = parts[1:]
            vote = []
            take_cands = max(1, int(len(ranking)*max_approval_percent))
            if take_cands > 0:
                vote = get_vote(take_cands, ranking)
                for cand in vote:
                    unique_candidates.add(cand)
            for _ in range(count):
                profile.append(copy(vote))

        used_candidate_count = len(unique_candidates)
        used_candidate_map = {}
        normalized_profile = []
        normalize_map = {}
        j = 0
        for i in unique_candidates:
            normalize_map[i] = j
            used_candidate_map[j] = candidate_map[i]
            j += 1
        for vote in profile:
            normalized_vote = []
            for c in vote:
                normalized_vote.append(normalize_map[c])
            normalized_profile.append(normalized_vote)
        if len(normalized_profile) != voter_count:
            raise Exception("Missing voters.")
        return used_candidate_map, normalized_profile, used_candidate_count
