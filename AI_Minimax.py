from copy import deepcopy
import numpy as np, math
from AI_Movement import free_cells, move
from AI_Heuristics import heuristics

# The Depth limit constant. You might change this if you want
# Keep in mind that your AI search might be pretty slow if you use too high depth
DEPTH = 4
MAX_VALUE = math.inf
MIN_VALUE = -math.inf


def maximize(grid, depth=DEPTH, alpha=MIN_VALUE, beta=MAX_VALUE):
    """
  Maximize function for the max (AI) of the MiniMax Algorithm
  If you want to change the depth of the search tree, try to
  implement some conditions for the "early stopping" at minimize
  or set up your own limit constant.
  """
    # TODO: Replace the value of the best_score
    # If you are not sure, check the implementation we talked about in week 2
    empty_cells = free_cells(grid)

    if depth == 0:
        return 0, heuristics(grid, len(empty_cells))

    best_score = MIN_VALUE
    best_move = None

    for i in range(4):

        moved_grid = deepcopy(grid)
        _, num, _ = move(moved_grid, i)

        if num != 0:
            new_score = minimize(moved_grid, depth - 1, alpha, beta)

            if best_score < new_score:
                best_move = i
                best_score = new_score

            # alpha = max(alpha, best_score)
            #
            # if beta <= alpha:
            #     break

    return best_move, best_score


# TODO: Implement maximize function here

def minimize(grid, depth=DEPTH, alpha=MIN_VALUE, beta=MAX_VALUE):
    """
    Minimize function for the min (Computer) of the Minimax Algorithm
    Computer put new 2 tile (with 90% probability) or
    4 tile (with 10% probability) at one of empty spaces
    """
    empty_cells = free_cells(grid)
    num_empty = len(empty_cells)

    if depth == 0:
        return heuristics(grid, num_empty)

    if num_empty == 0:
        # return heuristics(grid, num_empty)
        _, new_score = maximize(grid, depth - 1, alpha, beta)
        return new_score

    # TODO: (Optional) Implement conditions to stop the searching earlier
    # Would implement it after finish implementing Heuristics and MiniMax
    # ex) If there are enough empty spaces, we will proceed by skipping last two nodes
    if num_empty >= 6 and depth >= 3:
        return heuristics(grid, num_empty)

    sum_score = 0
    for c, r in empty_cells:
        for v in [2, 4]:
            new_grid = deepcopy(grid)
            new_grid[c][r] = v

            _, new_score = maximize(new_grid, depth - 1, alpha, beta)

            # beta = new_score
            # beta = min(beta, new_score)
            # if beta <= alpha:
            #     break

            if v == 2:
                new_score *= (0.9 / num_empty)  # Probability of 2 tile being placed
            else:
                new_score *= (0.1 / num_empty)  # Probability of 4 tile being placed

            sum_score += new_score

    return sum_score
