"""
Read and write data to files.

Two data formats are supported:
1. the Preflib format (soi, toi, soc, toc, or cat), and
2. .abc.yaml files (more expressive than Preflib files).
"""

import os
import ruamel.yaml
import preflibtools.instances as preflib
import pabutools.election as pb_election

from abcvoting.preferences import Profile, Voter
from abcvoting import misc


#: Valid keys for .abc.yaml files.
ABC_YAML_VALID_KEYS = [
    "profile",
    "num_cand",
    "committeesize",
    "compute",
    "voter_weights",
    "description",
]


class MalformattedFileException(Exception):
    """Malformatted file (Preflib or .abc.yaml)."""


def get_file_names(dir_name, filename_extensions=None):
    """
    List all file names in a directory that fit the specified filename extensions.

    .. important::

        Not recursive, i.e., does not look into sub-directories!

    Parameters
    ----------
        dir_name : str
            Path of directory to be searched for files.

        filename_extensions : list of str, optional
            File names must have one of these extensions.

    Returns
    -------
        list of str
            List of file names contained in the directory.
    """
    files = []
    for _, _, filenames in os.walk(dir_name):
        files = filenames
        break  # do not consider sub-directories
    if len(files) == 0:
        raise FileNotFoundError(f"No files found in {dir_name}")
    if filename_extensions:
        files = [
            f for f in files if any(f.endswith(extension) for extension in filename_extensions)
        ]
    return sorted(files)


def read_preflib_file(filename, top_ranks=None, setsize=None, use_weights=False):
    """
    Read a Preflib file (soi, toi, soc or toc).

    Parameters
    ----------
        filename : str
            Name of the Preflib file.

        top_ranks : int, default=1
            The approval set consists of the top entries in each voter's ranking in the Preflib
            instance. The parameter `top_ranks` determines how many ranks are joined to create
            the approval set. In case of of a Preflib category files, ranks correspond to
            categories.

            It cannot be used if parameter `setsize` is used too.

        setsize : int, optional
            Minimum number of candidates that voters approve.

            These candidates are taken from the top of ranking.
            In case of ties, more than setsize candidates are approved.

            It cannot be used if parameter `top_ranks` is used too.

        use_weights : bool, default=False
            Use weights of voters instead of individual voters.

            If False, treat vote count in Preflib file as the number of identical ballots,
            i.e., the number of voters that approve this set of candidates.
            If True, treat vote count as weight and use this weight in class Voter.

    Returns
    -------
        abcvoting.preferences.Profile
            Preference profile extracted from Preflib file.
    """
    if top_ranks is None and setsize is None:
        top_ranks = 1  # default
    if top_ranks and setsize:
        raise ValueError("Parameters top_ranks and setsize cannot be used simultaneously.")
    if top_ranks and top_ranks <= 0:
        raise ValueError("Parameter top_ranks must be > 0")
    if setsize and setsize <= 0:
        raise ValueError("Parameter setsize must be > 0")

    try:
        preflib_inst = preflib.get_parsed_instance(filename)
    except Exception as e:
        raise MalformattedFileException(
            "The preflib parser returned the following error: " + str(e)
        )

    if isinstance(preflib_inst, preflib.OrdinalInstance):
        if setsize:
            preflib_inst = preflib.CategoricalInstance.from_ordinal(
                preflib_inst, size_truncators=[setsize]
            )
        elif top_ranks:
            preflib_inst = preflib.CategoricalInstance.from_ordinal(
                preflib_inst, num_indif_classes=[1] * preflib_inst.num_alternatives
            )
    elif not isinstance(preflib_inst, preflib.CategoricalInstance):
        raise ValueError("Only ordinal and categorical preferences can be converted from PrefLib")

    # normalize candidates to 0, 1, 2, ...
    cand_names = []
    normalize_map = {}
    for cand, name in preflib_inst.alternatives_name.items():
        cand_names.append(name)
        normalize_map[cand] = len(cand_names) - 1

    profile = Profile(preflib_inst.num_alternatives, cand_names=cand_names)

    for preferences, count in preflib_inst.multiplicity.items():
        approval_set = []
        if setsize:
            category = 0
            while len(approval_set) < setsize and category < len(preferences):
                approval_set.extend([normalize_map[cand] for cand in preferences[category]])
                category += 1
            if 0 < len(approval_set) < setsize:
                approval_set = normalize_map.values()
        elif top_ranks:
            approval_set = [
                normalize_map[cand]
                for category in range(min(len(preferences), top_ranks))
                for cand in preferences[category]
            ]

        if use_weights:
            profile.add_voter(Voter(approval_set, weight=count))
        else:
            profile.add_voters([approval_set] * count)

    return profile


def read_preflib_files_from_dir(dir_name, top_ranks=None, setsize=None):
    """
    Read all Preflib files (soi, toi, soc or toc) in a given directory.

    Parameters
    ----------
        dir_name : str
            Path of the directory to be searched for Preflib files.

        top_ranks : int, default=1
            The approval set consists of the top entries in each voter's ranking in the Preflib
            instance. The parameter `top_ranks` determines how many ranks are joined to create
            the approval set. In case of of a Preflib category files, ranks correspond to
            categories.

            It cannot be used if parameter `setsize` is used too.

        setsize : int
            Minimum number of candidates that voters approve.

            These candidates are taken from the top of ranking.
            In case of ties, more than setsize candidates are approved.

            It cannot be used if parameter `top_ranks` is used too.

    Returns
    -------
        dict
            Dictionary with file names as keys and profiles (class abcvoting.preferences.Profile)
            as values.
    """
    files = get_file_names(dir_name, filename_extensions=[".soi", ".toi", ".soc", ".toc", ".cat"])

    profiles = {}
    for f in files:
        profile = read_preflib_file(
            os.path.join(dir_name, f), top_ranks=top_ranks, setsize=setsize
        )
        profiles[f] = profile
    return profiles


def write_profile_to_preflib_cat_file(filepath, profile):
    """
    Write a profile to a Preflib category file (.cat).

    Parameters
    ----------
        filepath : str
            File path of the Preflib file.

        profile : abcvoting.preferences.Profile
            Profile to be written.

    Returns
    -------
        None
    """
    preflib_inst = preflib.CategoricalInstance()
    preflib_inst.data_type = "cat"
    preflib_inst.num_categories = 2
    preflib_inst.categories_name = {"1": "Approved", "2": "Not approved"}
    preflib_inst.file_name = os.path.basename(filepath)
    preflib_inst.num_alternatives = profile.num_cand
    for cand in profile.candidates:
        preflib_inst.alternatives_name[cand + 1] = profile.cand_names[cand]

    for voter in profile:
        pref = (
            tuple(cand + 1 for cand in voter.approved),
            tuple(cand + 1 for cand in profile.candidates if cand not in voter.approved),
        )
        if int(voter.weight) == voter.weight:
            multiplicity = voter.weight
        else:
            multiplicity = 1
        if pref not in preflib_inst.preferences:
            preflib_inst.preferences.append(pref)
            preflib_inst.multiplicity[pref] = multiplicity
        else:
            preflib_inst.multiplicity[pref] += multiplicity
    preflib_inst.recompute_cardinality_param()
    preflib_inst.write(filepath)


def _yaml_flow_style_list(x):
    yamllist = ruamel.yaml.comments.CommentedSeq(x)
    yamllist.fa.set_flow_style()
    return yamllist


def read_abcvoting_yaml_file(filename):
    """
    Read contents of an abcvoting yaml file (ending with .abc.yaml).

    Parameters
    ----------
        filename : str
            File name of the .abc.yaml file.

    Returns
    -------
        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int or None
            The desired committee size.

        compute_instances : list of dict
            A list of compute instances, which are dictionaries.

            Compute instances can be passed to `Rule.compute`.

        data : dict
            The YAML data from `filename`.
    """
    yaml = ruamel.yaml.YAML(typ="safe", pure=True)
    with open(filename) as inputfile:
        data = yaml.load(inputfile)
    if "profile" not in data.keys():
        raise MalformattedFileException(f"{filename} does not contain a profile.")
    if "num_cand" in data.keys():
        num_cand = int(data["num_cand"])
    else:
        num_cand = max(cand for approval_set in data["profile"] for cand in approval_set) + 1
    profile = Profile(num_cand)
    approval_sets = data["profile"]
    if "voter_weights" in data.keys():
        weights = data["voter_weights"]
        if len(weights) != len(approval_sets):
            raise MalformattedFileException(
                f"{filename}: the number of voters differs from the number of voter weights."
            )
        for appr_set, weight in zip(approval_sets, weights):
            profile.add_voter(Voter(appr_set, weight=weight))
    else:
        profile.add_voters(approval_sets)

    if "committeesize" in data.keys():
        committeesize = int(data["committeesize"])
    else:
        committeesize = None

    if "compute" in data.keys():
        compute_instances = data["compute"]
    else:
        compute_instances = []
    for compute_instance in compute_instances:
        if "rule_id" not in compute_instance.keys():
            raise MalformattedFileException('Each rule instance (dict) requires key "rule_id".')
        compute_instance["profile"] = profile
        compute_instance["committeesize"] = committeesize
        if "result" in compute_instance.keys():
            if compute_instance["result"] is not None:
                # compute_instance["result"] should be a list of CandidateSet
                compute_instance["result"] = [
                    misc.CandidateSet(committee) for committee in compute_instance["result"]
                ]

    for key in data.keys():
        if key not in ABC_YAML_VALID_KEYS:
            raise MalformattedFileException(f'Key "{key}" is not valid (undefined).')

    return profile, committeesize, compute_instances, data


def write_abcvoting_instance_to_yaml_file(
    filename, profile, committeesize=None, compute_instances=None, description=None
):
    """
    Write abcvoting instance to an abcvoting yaml file.

    Parameters
    ----------
        filename : str
            File name of the .abc.yaml file.

        profile : abcvoting.preferences.Profile
            A profile.

        committeesize : int, optional
            The desired committee size.

        compute_instances : list of dict, optional
            A list of compute instances, which are dictionaries.

            Compute instances can be passed to `Rule.compute`.

        description : str, optional
            An optional description of the data.
    """
    data = {}
    if description is not None:
        data["description"] = description
    data["profile"] = _yaml_flow_style_list([list(voter.approved) for voter in profile])
    if not profile.has_unit_weights():
        data["voter_weights"] = _yaml_flow_style_list([voter.weight for voter in profile])
    data["num_cand"] = profile.num_cand
    if committeesize is not None:
        data["committeesize"] = committeesize
    modified_computed_instances = []
    if compute_instances is not None:
        for compute_instance in compute_instances:
            if "rule_id" not in compute_instance.keys():
                raise ValueError('Each compute instance (dict) requires key "rule_id".')
            mod_compute_instance = {"rule_id": compute_instance["rule_id"]}
            if "result" in compute_instance.keys():
                if compute_instance["result"] is None:
                    mod_compute_instance["result"] = None
                else:
                    mod_compute_instance["result"] = _yaml_flow_style_list(
                        [list(committee) for committee in compute_instance["result"]]
                    )  # TODO: would be nicer to store committees in set notation (curly braces)
            if "profile" in compute_instance.keys():  # this is superfluous information
                # check that the profile is the same as the main profile
                if str(compute_instance["profile"]) != str(profile):
                    raise ValueError(
                        "Compute instance contained a profile different from"
                        "the main profile passed to write_abcvoting_instance_to_yaml_file()."
                    )
            if "committeesize" in compute_instance.keys():  # this is superfluous information
                # check that the profile is the same as the main profile
                if int(compute_instance["committeesize"]) != committeesize:
                    raise ValueError(
                        "Compute instance contained a committee size different from"
                        "the committee size passed to write_abcvoting_instance_to_yaml_file()."
                    )
            for key in compute_instance.keys():
                # add other parameters to dictionary
                if key in ["rule_id", "result", "profile", "committeesize"]:
                    continue
                mod_compute_instance[key] = compute_instance[key]
            modified_computed_instances.append(mod_compute_instance)

        data["compute"] = modified_computed_instances

    yaml = ruamel.yaml.YAML()
    yaml.width = 120
    with open(filename, "w") as outfile:
        yaml.dump(data, outfile)


def pabutools_to_abcvoting(pb_instance, pb_profile):
    """
    Converts an instance and a profile from the pabutools into a profile for abcvoting.

    Parameters
    ----------
        pb_instance: pabutools.election.Instance
            The pabutools instance.
        pb_profile: pabutools.election.AbstractApprovalProfile
            The pabutools profile.

    Returns
    -------
        profile: abcvoting.preferences.Profile
            The abcvoting profile.
    """
    if not isinstance(pb_profile, pb_election.AbstractApprovalProfile):
        raise ValueError("Only approval profiles can be converted from the pabutools to abcvoting")

    num_cand = len(pb_instance)
    projects_to_cand = {p: i for i, p in enumerate(pb_instance)}
    profile = Profile(num_cand=num_cand, cand_names=[p.name for p in pb_instance])

    for pb_ballot in pb_profile:
        for _ in range(pb_profile.multiplicity(pb_ballot)):
            profile.add_voter(Voter([projects_to_cand[a] for a in pb_ballot], num_cand=num_cand))

    return profile


def read_pabulib_file(file_path):
    """
    Read the content of a pabulib file and returns the corresponding profile.
    All projects are assumed to have the same cost and the size of the committee is equal to
    the budget limit of the instance (this does not always make sense).

    Parameters
    ----------
        file_path: str
            The file path.

    Returns
    -------
        profile: abcvoting.preferences.Profile
            The profile.
        committeesize: int
            The size of the committee.
    """
    instance, pb_profile = pb_election.parse_pabulib(file_path)
    return pabutools_to_abcvoting(instance, pb_profile), int(instance.budget_limit)


def abcvoting_to_pabutools(profile: Profile, committeesize: int):
    """
    Converts an abcvoting profile to a pabutools profile.

    Parameters
    ----------
        profile: Profile
            The abcvoting profile.
        committeesize: int
            The size of the committee.

    Returns
    -------
        instance: pabutools.election.Instance
            The pabutools instance.
        profile: pabutools.election.ApprovalProfile
            The pabutools approval profile.
    """

    pb_instance = pb_election.Instance()
    pb_instance.budget_limit = committeesize
    alt_to_project = {}
    for alt in profile.candidates:
        p = pb_election.Project(alt, cost=1)
        pb_instance.add(p)
        alt_to_project[alt] = p
    pb_profile = pb_election.ApprovalProfile()
    for voter in profile:
        pb_profile.append(pb_election.ApprovalBallot([alt_to_project[a] for a in voter.approved]))
    return pb_instance, pb_profile


def write_pabulib_file(profile, committeesize, file_path):
    """
    Write an abcvoting instance as a pabulib file.

    Parameters
    ----------
        file_path: str
            The file path.
        profile: Profile
            The abcvoting profile.
        committeesize: int
            The size of the committee.
    """
    pb_instance, pb_profile = abcvoting_to_pabutools(profile, committeesize)
    print(pb_instance)
    print(pb_profile)
    pb_election.write_pabulib(pb_instance, pb_profile, file_path)
