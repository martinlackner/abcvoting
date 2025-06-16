"""
Unit tests for abcvoting/fileio.py.
"""

import pytest
from abcvoting import fileio, abcrules
import os
from abcvoting.preferences import Profile, Voter


@pytest.mark.parametrize("filename", ["test1.toi", "test2.soi", "test3.toc"])
def test_readfile(filename):
    currdir = os.path.dirname(os.path.abspath(__file__))
    profile = fileio.read_preflib_file(currdir + "/data/" + filename, top_ranks=2)
    assert len(profile) == 5
    assert profile.has_unit_weights()


def test_readfromdir():
    currdir = os.path.dirname(os.path.abspath(__file__))
    profiles = fileio.read_preflib_files_from_dir(currdir + "/data/", setsize=2)
    assert len(profiles) == 5
    for filename, profile in profiles.items():
        assert isinstance(filename, str)
        for voter in profile:
            assert len(voter.approved) >= 2
        assert profile.has_unit_weights()


@pytest.mark.parametrize("filename", ["test2.soi", "test3.toc"])
def test_readfile_setsize(filename):
    currdir = os.path.dirname(os.path.abspath(__file__))
    profile = fileio.read_preflib_file(currdir + "/data/" + filename, setsize=2)
    for voter in profile:
        assert len(voter.approved) == 2


@pytest.mark.parametrize("filename", ["test2.soi", "test3.toc"])
def test_readfile_numcats(filename):
    currdir = os.path.dirname(os.path.abspath(__file__))
    profile = fileio.read_preflib_file(currdir + "/data/" + filename, top_ranks=2)
    for voter in profile:
        assert len(voter.approved) == 2


@pytest.mark.parametrize(
    "filename,setsize,expected",
    [
        ("test1.toi", 1, [1, 1, 1, 2, 1]),
        ("test1.toi", 2, [2, 3, 3, 2, 2]),
        ("test1.toi", 3, [3, 3, 3, 3, 3]),
        ("test1.toi", 4, [6, 6, 6, 6, 4]),
        ("test1.toi", 5, [6, 6, 6, 6, 6]),
        ("test4.cat", 1, [1, 1, 1, 3, 1]),
        ("test4.cat", 2, [2, 6, 6, 3, 2]),
        ("test4.cat", 3, [3, 6, 6, 3, 3]),
        ("test4.cat", 4, [6, 6, 6, 6, 6]),
    ],
)
def test_readfile_setsize_with_ties(filename, setsize, expected):
    currdir = os.path.dirname(os.path.abspath(__file__))
    profile = fileio.read_preflib_file(currdir + "/data/" + filename, setsize=setsize)
    assert [len(voter.approved) for voter in profile] == expected
    for voter in profile:
        assert voter.weight == 1


@pytest.mark.parametrize(
    "filename,top_ranks,expected",
    [
        ("test1.toi", 1, [1, 1, 1, 2, 1]),
        ("test1.toi", 2, [2, 3, 3, 3, 2]),
        ("test1.toi", 3, [3, 3, 3, 3, 3]),
        ("test1.toi", 4, [3, 3, 3, 3, 4]),
        ("test1.toi", 5, [3, 3, 3, 3, 4]),
        ("test4.cat", 1, [1, 1, 1, 3, 1]),
        ("test4.cat", 2, [2, 1, 1, 3, 1]),
        ("test4.cat", 3, [2, 1, 1, 3, 2]),
        ("test4.cat", 4, [3, 1, 1, 3, 3]),
    ],
)
def test_readfile_top_ranks_with_ties(filename, top_ranks, expected):
    currdir = os.path.dirname(os.path.abspath(__file__))
    profile = fileio.read_preflib_file(currdir + "/data/" + filename, top_ranks=top_ranks)
    assert [len(voter.approved) for voter in profile] == expected
    for voter in profile:
        assert voter.weight == 1


@pytest.mark.parametrize(
    "filename", ["test1.toi", "test2.soi", "test3.toc", "test6.soi", "test7.toi", "test8.soi"]
)
def test_readfile_corrupt(filename):
    currdir = os.path.dirname(os.path.abspath(__file__))
    with pytest.raises(fileio.MalformattedFileException):
        profile = fileio.read_preflib_file(currdir + "/data-fail/" + filename, setsize=2)
        print(str(profile))


@pytest.mark.parametrize(
    "filename,total_weight,num_voters",
    [("test1.toi", 5, 4), ("test2.soi", 5, 3), ("test3.toc", 5, 5), ("test4.cat", 5, 4)],
)
def test_readfile_and_weights(filename, total_weight, num_voters):
    currdir = os.path.dirname(os.path.abspath(__file__))
    profile = fileio.read_preflib_file(currdir + "/data/" + filename)
    assert len(profile) == total_weight
    for voter in profile:
        assert voter.weight == 1
    profile = fileio.read_preflib_file(currdir + "/data/" + filename, use_weights=True)
    assert len(profile) == num_voters
    assert sum(voter.weight for voter in profile) == total_weight


def test_read_and_write_preflib_file():
    currdir = os.path.dirname(os.path.abspath(__file__))
    profile1 = Profile(6)
    profile1.add_voters([[3], [4, 1, 5], [0, 2], [], [0, 1, 2, 3, 4, 5], [5], [1], [1, 2]])
    fileio.write_profile_to_preflib_cat_file(currdir + "/data/test5.cat", profile1)
    for use_weights in [True, False]:
        profile2 = fileio.read_preflib_file(currdir + "/data/test5.cat", use_weights=use_weights)
        assert len(profile1) == len(profile2)
        for i, voter in enumerate(profile1):
            assert voter.weight == profile2[i].weight
            assert voter.approved == set(profile2[i].approved)


def test_read_and_write_abc_yaml_file():
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = currdir + "/data/test6.abc.yaml"
    profile1 = Profile(6)
    profile1.add_voters([[3], [4, 1, 5], [0, 2], [], [0, 1, 2, 3, 4, 5], [5], [1], [1]])
    committeesize1 = 3
    compute_instances1 = [
        {"rule_id": "pav", "resolute": True},
        {"rule_id": "seqphragmen", "algorithm": "float-fractions"},
        {
            "rule_id": "equal-shares",
            "algorithm": "standard-fractions",
            "completion": None,
        },
    ]

    fileio.write_abcvoting_instance_to_yaml_file(
        filename, profile1, committeesize=committeesize1, compute_instances=compute_instances1
    )

    profile2, committeesize2, compute_instances2, _ = fileio.read_abcvoting_yaml_file(filename)
    assert committeesize1 == committeesize2
    assert len(profile1) == len(profile2)
    for i, voter in enumerate(profile1):
        assert voter.weight == profile2[i].weight
        assert voter.approved == set(profile2[i].approved)
    for i in range(len(compute_instances1)):
        assert abcrules.compute(
            profile=profile1, committeesize=committeesize1, **compute_instances1[i]
        ) == abcrules.compute(**compute_instances2[i])
    for compute_instance in compute_instances2:
        compute_instance.pop("profile")
        compute_instance.pop("committeesize")
    assert compute_instances1 == compute_instances2


def test_read_special_abc_yaml_file1():
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = currdir + "/data/test7.abc.yaml"

    profile1 = Profile(6)
    profile1.add_voter(Voter([3], weight=1.3))
    profile1.add_voters([[4, 1, 5], [0, 2], [], [0, 1, 2, 3, 4, 5], [5]])
    profile1.add_voter(Voter([1], weight=2))
    fileio.write_abcvoting_instance_to_yaml_file(filename, profile1, description="just a profile")

    profile2, committeesize, compute_instances2, data2 = fileio.read_abcvoting_yaml_file(filename)
    assert str(profile1) == str(profile2)
    assert committeesize is None
    assert compute_instances2 == []


def test_read_special_abc_yaml_file2():
    currdir = os.path.dirname(os.path.abspath(__file__))
    filename = currdir + "/data/test8.abc.yaml"

    profile1 = Profile(6)
    profile1.add_voters([{3}, {1, 4, 5}, {0, 2}, {}, {0, 1, 2, 3, 4, 5}, {1, 3}, {1}, {1}])

    profile2, committeesize, compute_instances, data = fileio.read_abcvoting_yaml_file(filename)
    assert str(profile1) == str(profile2)
    assert committeesize == 2
    assert len(compute_instances) == 1
    assert abcrules.compute(**compute_instances[0]) == [{1, 3}]


@pytest.mark.parametrize("filename", ["test1.abc.yaml", "test2.abc.yaml", "test3.abc.yaml"])
def test_read_corrupt_abc_yaml_file(filename):
    currdir = os.path.dirname(os.path.abspath(__file__))
    with pytest.raises(fileio.MalformattedFileException):
        profile = fileio.read_abcvoting_yaml_file(currdir + "/data-fail/" + filename)
        print(str(profile))


@pytest.mark.parametrize(
    "filename",
    [
        "poland_lodz_2020_mileszki.pb",
        "poland_lodz_2020_nad-nerem.pb",
        "poland_lodz_2020_nowosolna.pb",
    ],
)
def test_read_and_write_pabulib(filename):
    currdir = os.path.dirname(os.path.abspath(__file__))
    profile, committee_size = fileio.read_pabulib_file(currdir + "/data/" + filename)
    fileio.write_pabulib_file(profile, committee_size, "tmp.pb")
    os.remove("tmp.pb")
