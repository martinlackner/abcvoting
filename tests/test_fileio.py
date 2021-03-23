"""
Unit tests for fileio.py
"""

import pytest
from abcvoting import fileio
import os
from abcvoting.preferences import Profile


@pytest.mark.parametrize("filename", ["test1.toi", "test2.soi", "test3.toc"])
def test_readfile(filename):
    currdir = os.path.dirname(os.path.abspath(__file__))
    profile = fileio.read_preflib_file(currdir + "/data/" + filename, relative_setsize=0.5)
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


@pytest.mark.parametrize(
    "filename,setsize,expected",
    [
        ("test1.toi", 1, [1, 1, 1, 2, 1]),
        ("test1.toi", 2, [2, 3, 3, 2, 2]),
        ("test1.toi", 3, [3, 3, 3, 3, 3]),
        ("test1.toi", 4, [6, 6, 6, 6, 4]),
        ("test1.toi", 5, [6, 6, 6, 6, 6]),
        ("test4.toi", 1, [1, 1, 1, 3, 1]),
        ("test4.toi", 2, [2, 6, 6, 3, 2]),
        ("test4.toi", 3, [3, 6, 6, 3, 3]),
        ("test4.toi", 4, [6, 6, 6, 6, 6]),
    ],
)
def test_readfile_setsize_with_ties(filename, setsize, expected):
    currdir = os.path.dirname(os.path.abspath(__file__))
    profile = fileio.read_preflib_file(currdir + "/data/" + filename, setsize=setsize)
    assert [len(voter.approved) for voter in profile] == expected
    for voter in profile:
        assert voter.weight == 1


@pytest.mark.parametrize(
    "filename", ["test1.toi", "test2.soi", "test3.toc", "test6.soi", "test7.toi", "test8.soi"]
)
def test_readfile_corrupt(filename):
    currdir = os.path.dirname(os.path.abspath(__file__))
    with pytest.raises(fileio.PreflibException):
        profile = fileio.read_preflib_file(currdir + "/data-fail/" + filename, setsize=2)
        print(str(profile))


@pytest.mark.parametrize(
    "filename,total_weight,num_voters",
    [("test1.toi", 5, 4), ("test2.soi", 5, 3), ("test3.toc", 5, 5), ("test4.toi", 5, 4)],
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
    profile1.add_voters([[3], [4, 1, 5], [0, 2], [], [0, 1, 2, 3, 4, 5], [5], [1], [1]])
    fileio.write_profile_to_preflib_toi_file(profile1, currdir + "/data/test5.toi")
    for use_weights in [True, False]:
        profile2 = fileio.read_preflib_file(currdir + "/data/test5.toi", use_weights=use_weights)
        assert len(profile1) == len(profile2)
        for i, voter in enumerate(profile1):
            assert voter.weight == profile2[i].weight
            assert voter.approved == set(profile2[i].approved)
