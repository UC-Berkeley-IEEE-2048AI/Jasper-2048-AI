import math
import os
import pygame
from copy import deepcopy
from pprint import pprint

import numpy as np
from _2048.game import Game2048
from _2048.manager import GameManager

from AI_Minimax import maximize

MAX_VALUE = math.inf
MIN_VALUE = -math.inf

# Exported from the package: Creates the events using pygame key
EVENTS = [
    pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_UP}),  # UP
    pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_RIGHT}),  # RIGHT
    pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_DOWN}),  # DOWN
    pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_LEFT})  # LEFT
]

''' 
  This is a 2D-List for cells in order to check for each action
  (0,0) is the upper left corner
  ex) for CELLS[0] (up action), since we need to check the top of each column
      so that we can merge/move tiles towards the top
      [(0, 0), (0, 1), (0, 2), (0, 3), 
      (1, 0), (1, 1), (1, 2), (1, 3), 
      (2, 0), (2, 1), (2, 2), (2, 3), 
      (3, 0), (3, 1), (3, 2), (3, 3)]
'''
CELLS = [
    [(r, c) for r in range(4) for c in range(4)],  # UP:
    [(r, c) for r in range(4) for c in range(4 - 1, -1, -1)],  # RIGHT
    [(r, c) for r in range(4 - 1, -1, -1) for c in range(4)],  # DOWN
    [(r, c) for r in range(4) for c in range(4)],  # LEFT
]

'''
  Generator for the possible movement from the current cell
  For up/down event, will track the colum. For left/right, will track the row
  ex) if we do GET_DELTAS[0](0,0) (move up to the (0,0)), this will generate (1,0) -> (2,0) -> (3,0)
      because we would check the move/merge of the closest tile from (0,0) first.
  ex) if we do GET_DELTAS[1](1,0) (move right to (1,0)), this will generate nothing (since it's at the leftmost column)
'''
GET_DELTAS = [
    lambda r, c: ((i, c) for i in range(r + 1, 4)),  # UP
    lambda r, c: ((r, i) for i in range(c - 1, -1, -1)),  # RIGHT
    lambda r, c: ((i, c) for i in range(r - 1, -1, -1)),  # DOWN
    lambda r, c: ((r, i) for i in range(c + 1, 4))  # LEFT
]


def run_game(game_class=Game2048, title='2048!', data_dir='save'):
    '''
  This function will run the 2048 game for one-AI-player 
  '''

    # Initialize game using py_game package
    pygame.init()
    pygame.display.set_caption(title)
    pygame.display.set_icon(game_class.icon(32))
    clock = pygame.time.Clock()

    # Created the directory to save the gmae grid and max score
    os.makedirs(data_dir, exist_ok=True)

    screen = pygame.display.set_mode((game_class.WIDTH, game_class.HEIGHT))

    # screen = pygame.display.set_mode((50, 20))
    manager = GameManager(Game2048, screen,
                          os.path.join(data_dir, '2048.score'),
                          os.path.join(data_dir, '2048.%d.state'))

    # This will faster the animation
    manager.game.ANIMATION_FRAMES = 1
    manager.game.WIN_TILE = 999999

    tick = 0
    running = True
    # 2048 Game loop
    while running:
        clock.tick(30)
        tick += 1

        if tick % 5 == 0:
            old_grid = deepcopy(manager.game.grid)

            # Use Minimax algorithm to decide best possible move
            best_move, best_score = maximize(old_grid, alpha=MIN_VALUE)

            if best_move is None:
                print('No way! Maximum number is %s' % np.max(manager.game.grid))
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONUP:
                        manager.dispatch(event)
            else:

                print("Best move: ", move_name(best_move))
                print("Move Score: ", best_score)
                e = EVENTS[best_move]
                manager.dispatch(e)
                # move(manager.game.grid, best_move)
                pprint(manager.game.grid, width=30)
                print()
                # print(manager.game.score)

        for event in pygame.event.get():
          if event.type == pygame.QUIT:
            running = False
          elif event.type == pygame.MOUSEBUTTONUP:
            manager.dispatch(event)

        manager.draw()

    pygame.quit()
    manager.close()


# Start game
def move_name(move):
    if move == 0:
        return "Up"
    elif move == 1:
        return "Right"
    elif move == 2:
        return "Down"
    else:
        return "Left"

if __name__ == "__main__":
    run_game()


