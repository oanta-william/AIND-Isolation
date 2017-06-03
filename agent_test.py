"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import cProfile

# from importlib import reload

from isolation import *
import game_agent
from sample_players import *
from game_agent import *

class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        # reload(game_agent)
        self.player1 = "Player1"
        self.player2 = "Player2"
        self.game = isolation.Board(self.player1, self.player2)

    def testOneMatch(self):
        player1 = AlphaBetaPlayer()
        player2 = MinimaxPlayer()

        game = Board(player1, player2)
        winner, history, outcome = game.play()

        print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
        print(game.to_string())
        print("Move history:\n{!s}".format(history))

if __name__ == '__main__':
    unittest.main()
