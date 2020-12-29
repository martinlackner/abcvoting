"""
Unit tests for misc.py
"""

import pytest
from abcvoting import misc


@pytest.mark.parametrize(
    "a, b, dist", [([1, 2, 3], [1, 3, 4], 2), ([1, 2, 3], [0, 4], 5),
                   ([1, 2], [2, 1], 0), ([0, 1, 2, 3], [2, 3, 4, 5], 4)]
)
def test_hamming(a, b, dist):
    assert misc.hamming(a, b) == dist
