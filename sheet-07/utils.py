"""
Copyright 2019, University of Freiburg
Chair of Algorithms and Data Structures.
Patrick Brosi <brosi@cs.uni-freiburg.de>
Sebastian Walter <swalter@cs.uni-freiburg.de>
"""

import itertools
from array import array


def ped(x: str, y: str, delta: int) -> int:
    """

    Computes the prefix edit distance PED(x,y) for the two given strings x and
    y. Returns PED(x,y) if it is smaller or equal to the given delta; delta + 1
    otherwise.

    This is the Python version. To use the (much faster) Rust version, install
    the package ad-freiburg-qgram-utils with pip.

    >>> ped("frei", "frei", 0)
    0
    >>> ped("frei", "freiburg", 0)
    0
    >>> ped("frei", "breifurg", 4)
    1
    >>> ped("fr", "tifurg", 2)
    2
    >>> ped("freiburg", "stuttgart", 2)
    3
    >>> ped("", "freiburg", 10)
    0
    >>> ped("", "", 10)
    0
    """

    # Compute the dimensions of the matrix.
    n = len(x) + 1
    # Note that it is enough to compute the first |x| + δ + 1 columns.
    m = min(n + delta, len(y) + 1)

    matrix = array("I", [0] * m * n)

    # Initialize the first column.
    for row in range(n):
        matrix[m * row] = row
    # Initialize the first row.
    for i in range(m):
        matrix[i] = i

    # Compute the rest of the matrix.
    for row in range(1, n):
        for col in range(1, m):
            s = 1
            if x[row - 1] == y[col - 1]:
                s = 0

            rep_costs = matrix[m * (row - 1) + (col - 1)] + s
            add_costs = matrix[m * row + (col - 1)] + 1
            del_costs = matrix[m * (row - 1) + col] + 1

            matrix[m * row + col] = min(rep_costs, add_costs, del_costs)

    # Search the last row for the minimum value.
    delta_min = delta + 1
    for col in range(m):
        val = matrix[m * (n - 1) + col]
        if val < delta_min:
            delta_min = matrix[m * (n - 1) + col]

    return delta_min


InvertedList = list[tuple[int, int]]


def merge_lists(lists: list[InvertedList]) -> InvertedList:
    """

    Merges the given inverted lists, where each list
    contains (ID, freq) tuples.

    >>> merge_lists([[(1, 2), (3, 1), (5, 1)], [(2, 1), (3, 2), (9, 2)]])
    [(1, 2), (2, 1), (3, 3), (5, 1), (9, 2)]
    >>> merge_lists([[(1, 2), (3, 1), (5, 1)], []])
    [(1, 2), (3, 1), (5, 1)]
    >>> merge_lists([[], []])
    []
    """
    merged: InvertedList = []
    for el in sorted(itertools.chain.from_iterable(lists)):
        if len(merged) != 0 and merged[-1][0] == el[0]:
            merged[-1] = (merged[-1][0], merged[-1][1] + el[1])
        else:
            merged.append(el)
    return merged
