import random

import numpy as np, math
from copy import deepcopy

COUNT_X = 4
COUNT_Y = 4

pattern = [[0, 0, 1, 3],
           [0, 1, 3, 5],
           [1, 3, 5, 15],
           [3, 5, 15, 30]]

corner = [[0.0, 0.0, 0.1, 0.1],
          [0.0, 0.1, 0.1, 0.3],
          [0.1, 0.1, 0.3, 0.5],
          [0.1, 0.3, 0.5, 1]]


def heuristics(grid, num_empty):
    """
    This function scores the grid based on the algorithm implemented
    so that the maximize function of AI_Minimax can decide which branch
    to follow.
    """
    grid = np.array(grid)
    score = 0


    # TODO: Implement your heuristics here.
    # You are more than welcome to implement multiple heuristics
    p_score = pattern_score(grid)
    m_score = mono_score(grid)
    smooth_score = smoothness(grid)
    corner = largest_in_corner(grid)
    # Weight for each score

    # Weights
    empty_weight = 200
    mono_weight = 100
    smooth_weight = 1
    pattern_weight = 0.5
    corner_weight = 1500

    # scoring
    # score += p_score * pattern_weight
    # score += (np.sum(grid ** 2)//2)
    score += m_score * mono_weight
    score += smooth_score * smooth_weight
    score += corner * corner_weight
    score += num_empty * empty_weight

    return score


def smoothness(grid):
    score = 0
    # # Check cols
    score -= np.sum(abs(grid[:, 0] - grid[:, 1]))
    score -= np.sum(abs(grid[:, 1] - grid[:, 2]))
    score -= np.sum(abs(grid[:, 2] - grid[:, 3]))

    # # Check rows
    score -= np.sum(abs(grid[0, :] - grid[1, :]))
    score -= np.sum(abs(grid[1, :] - grid[2, :]))
    score -= np.sum(abs(grid[2, :] - grid[3, :]))

    return score


def mono_score(grid):

    # Ensure the rows are either increasing or decreasing
    colsTopBottom = sum([is_increasing(grid[:, i]) for i in range(0, COUNT_X)])
    colsBottomTop = sum([is_decreasing(grid[:, i]) for i in range(0, COUNT_X)])

    rowsLeftRight = sum([is_increasing(grid[i, :]) for i in range(0, COUNT_X)])
    rowsRightLeft = sum([is_decreasing(grid[i, :]) for i in range(0, COUNT_X)])

    BRCorner = colsTopBottom + rowsLeftRight
    TRCorner = colsBottomTop + rowsLeftRight

    BLCorner = colsTopBottom + rowsRightLeft
    TLCorner = colsBottomTop + rowsRightLeft

    # # or TRCorner or BLCorner or TLCorner
    return BRCorner


def is_increasing(arr):
    last = arr[0]
    for i in range(1, COUNT_X):
        if last > arr[i]:
            return False
        last = arr[i]
    return True


def is_decreasing(arr):
    last = arr[0]
    for i in range(1, COUNT_X):
        if last < arr[i]:
            return False
        last = arr[i]
    return True


def pattern_score(grid):
    score = 0
    for x in range(4):
        for y in range(4):
            score += grid[x][y] * pattern[x][y]

    return score


"""
Helper Functions
"""


def row_is_montonic(grid):
    rows = np.all(grid[:, 1:] >= grid[:, :-1], axis=1)
    mono_rows = np.count_nonzero(rows)
    return mono_rows, mono_rows != 0


def column_is_montonic(grid):
    cols = np.all(grid[:, 1:] >= grid[:, :-1], axis=0)
    mono_cols = np.count_nonzero(cols)
    return mono_cols, mono_cols != 0


def get_largest_value(grid):
    max_value = 0

    for x in range(COUNT_X):
        for y in range(COUNT_Y):
            if max_value < grid[y][x]:
                max_value = grid[y][x]

    return max_value


def largest_in_corner(grid):
    largest = get_largest_value(grid)

    if grid[0][0] == largest:
        return True

    if grid[0][3] == largest:
        return True

    if grid[3][0] == largest:
        return True

    if grid[3][3] == largest:
        return True
    return False
