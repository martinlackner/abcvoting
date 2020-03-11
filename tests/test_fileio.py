"""
Unit tests for fileio.py
"""

import pytest
from abcvoting import fileio
import os


@pytest.mark.parametrize(
    "filename", ["test1.toi", "test2.soi", "test3.toc"]
)
def test_readfile(filename):
    currdir = os.path.dirname(os.path.abspath(__file__))
    profile = fileio.read_preflib_file(
        currdir + "/data/" + filename, appr_percent=0.5)
    assert len(profile) == 5
    assert profile.has_unit_weights()


def test_readfromdir():
    currdir = os.path.dirname(os.path.abspath(__file__))
    profiles = \
        fileio.load_preflib_files_from_dir(currdir + "/data/",
                                           setsize=2)
    assert len(profiles) == 4
    for profile in profiles:
        assert len(profile) == 5
        for pref in profile:
            assert len(pref) >= 2
        assert profile.has_unit_weights()


@pytest.mark.parametrize(
    "filename", ["test2.soi", "test3.toc"]
)
def test_readfile_setsize(filename):
    currdir = os.path.dirname(os.path.abspath(__file__))
    profile = \
        fileio.read_preflib_file(currdir + "/data/" + filename,
                                 setsize=2)
    for pref in profile:
        assert len(pref) == 2


@pytest.mark.parametrize(
    "filename,setsize,expected", 
    [("test1.toi", 1, [1, 1, 1, 2, 1]),
     ("test1.toi", 2, [2, 3, 3, 2, 2]),
     ("test1.toi", 3, [3, 3, 3, 3, 3]),
     ("test1.toi", 4, [6, 6, 6, 6, 4]),
     ("test1.toi", 5, [6, 6, 6, 6, 6]),
     ("test4.toi", 1, [1, 1, 1, 3, 1]),
     ("test4.toi", 2, [2, 6, 6, 3, 2]),
     ("test4.toi", 3, [3, 6, 6, 3, 3]),
     ("test4.toi", 4, [6, 6, 6, 6, 6])]
)
def test_readfile_setsize_with_ties(filename, setsize, expected):
    currdir = os.path.dirname(os.path.abspath(__file__))
    profile = \
        fileio.read_preflib_file(currdir + "/data/" + filename,
                                 setsize=setsize)
    assert [len(pref) for pref in profile] == expected


@pytest.mark.parametrize(
    "filename", ["test1.toi", "test2.soi", "test3.toc",
                 "test6.soi", "test7.toi", "test8.soi"]
)
def test_readfile_corrupt(filename):
    currdir = os.path.dirname(os.path.abspath(__file__))
    with pytest.raises(fileio.PreflibException):
        profile = fileio.read_preflib_file(
            currdir + "/data-fail/" + filename, setsize=2)
        print(str(profile))
