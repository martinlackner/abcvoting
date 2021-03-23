"""
Unit tests for misc.py
"""

import pytest
from abcvoting import misc


@pytest.mark.parametrize(
    "a, b, dist",
    [
        ([1, 2, 3], [1, 3, 4], 2),
        ([1, 2, 3], [0, 4], 5),
        ([1, 2], [2, 1], 0),
        ([0, 1, 2, 3], [2, 3, 4, 5], 4),
    ],
)
def test_hamming(a, b, dist):
    assert misc.hamming(a, b) == dist


def test_compare_list_of_committees():
    committees1 = [{1, 2}, {3, 4}, {0, 3}]
    committees2 = [{3, 4}, {3, 0}, {2, 1}]
    assert misc.compare_list_of_committees(committees1, committees2)
    committees1[0].update([1])
    assert misc.compare_list_of_committees(committees1, committees2)
    committees1.append({1, 2, 3})
    assert not misc.compare_list_of_committees(committees1, committees2)
    committees1 = [{1, 2}, {3}, {0, 3}]
    assert not misc.compare_list_of_committees(committees1, committees2)


def test_str_committees_header():
    assert misc.str_committees_header([{1, 0}]) == "1 committee:"
    assert misc.str_committees_header([{1, 0}, {1, 2, 3}]) == "2 committees:"
    assert misc.str_committees_header([]) == "No committees"
    assert misc.str_committees_header([{1, 0}], winning=True) == "1 winning committee:"
    assert misc.str_committees_header([{1, 0}, {1, 2, 3}], winning=True) == "2 winning committees:"
    assert (
        misc.str_committees_header([], winning=True)
        == "No winning committees (this should not happen)"
    )
