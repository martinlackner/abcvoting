"""
Unit tests for fileio.py
"""

import pytest
from abcvoting import fileio
from abcvoting.preferences import Profile
import os


@pytest.mark.parametrize(
    "filename", ["test1.toi", "test2.soi"]
)
def test_readfile(filename):
    currdir = os.path.dirname(os.path.abspath(__file__))
    _, appr_sets, num_cand = \
        fileio.read_election_file(currdir + "/data/" + filename,
                                  max_approval_percent=0.5)
    assert len(appr_sets) == 5
    profile = Profile(num_cand)
    profile.add_preferences(appr_sets)
    assert profile.has_unit_weights()
