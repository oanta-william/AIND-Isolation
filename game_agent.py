"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import numpy as np


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # if board_state[3][3] != 0:
    #     if (board_state.transpose() == board_state).all():
    #         # game.

    # if game.move_count < 4:
    return float(len(game.get_legal_moves(player))) - float(len(game.get_legal_moves(game.get_opponent(player))))

    # board_state = np.array(game._board_state)
    #
    # moves = game.get_legal_moves(player)
    #
    # board_state_matrix = board_state[:-3].reshape((7, 7))
    #
    # add_value = 0
    # for m in moves:
    #
    #     add_value += len(game.forecast_move(m).get_legal_moves(game.inactive_player))
    #
    #     # Compute possible legal moves mask centered at (a, b)
    #     a, b = m
    #     length = 7
    #     radius = 2
    #
    #     y, x = np.ogrid[-a:length - a, -b:length - b]
    #     next_move_mask = x * x + y * y == radius ** 2 + 1
    #
    #     legal_moves = np.sum(board_state_matrix[next_move_mask]) / 2
    #
    #     add_value += legal_moves


    #
    # -14 + -1 = -15
    #
    # -2 * 7 + 1 = -13
    #
    # -1 * 7 + -2 = - 9
    #
    # -1 * 7 + 2 = - 5
    #
    # 1 * 7 + -2 = 5
    #
    # 1 * 7 + 2 = 9
    #
    # 2 * 7 + -1 = 13
    #
    # 2 * 7 + 1 = 15

    #
    # directions = [[-2, -1], [-2, 1], [-1, -2], [-1, 2],
    #               [1, -2], [1, 2], [2, -1], [2, 1]]

    #
    # legal_moves = np.array(game.get_legal_moves())
    # board = np.array(game._board_state)
    #
    # directions = np.array([-15, -13, -9, -5, 5, 9, 13, 15])
    #
    # add_value = 0
    # for m in legal_moves:
    #     m_prime = np.array(m[0] * 7 + m[1])
    #
    #     new_directions = directions + m_prime
    #
    #     new_directions = new_directions[new_directions >= 0]
    #     new_directions = new_directions[new_directions <= 49]
    #     # new_directions = new_directions[new_directions >= 0 and new_directions <= 49]
    #     add_value += np.sum(board[new_directions])

    # if game.is_loser(player):
    #     return float("-inf")
    #
    # if game.is_winner(player):
    #     return float("inf")
    #
    # value = float(len(legal_moves)) - float(len(game.get_legal_moves(game.get_opponent(player)))) - float(add_value)
    #
    # return value


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return float(len(game.get_legal_moves(player))) - float(len(game.get_legal_moves(game.get_opponent(player))))


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return float(len(game.get_legal_moves(player))) - float(len(game.get_legal_moves(game.get_opponent(player))))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=2, score_fn=custom_score_2, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.boards = dict()
        self.moves = dict()


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        def max_value(game, current_depth):

            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if current_depth == 0 or len(game.get_legal_moves()) < 1:
                return self.score(game, self)

            v = float("-inf")
            for a in game.get_legal_moves():
                v = max(v, min_value(game.forecast_move(a), current_depth - 1))
            return v

        def min_value(game, current_depth):

            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if current_depth == 0 or len(game.get_legal_moves()) < 1:
                return self.score(game, self)

            v = float("inf")

            for a in game.get_legal_moves():
                v = min(v, max_value(game.forecast_move(a), current_depth - 1))
            return v

        best_move = None
        best_val = float("-inf")

        for a in game.get_legal_moves():
            v = min_value(game.forecast_move(a), depth - 1)
            if v > best_val:
                best_move = a
                best_val = v
        return best_move


class NegamaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited negamax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move = self.negamax(game, self.search_depth)
            return best_move

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def negamax(self, game, depth):
        """Implement depth-limited negamax search algorithm as described in
        the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        def negamax(game, depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0 or len(game.get_legal_moves()) < 1:
                return None, self.score(game, game.active_player)

            best_move = None
            best_val = float("-inf")

            for a in game.get_legal_moves():
                _, v = negamax(game.forecast_move(a), depth - 1)
                current_score = -v

                if current_score > best_val:
                    best_move = a
                    best_val = current_score
            return best_move, best_val

        best_move, _ = negamax(game, depth)

        return best_move


class ABNegamaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited negamax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move = self.ab_negamax(game, self.search_depth)
            return best_move

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def ab_negamax(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited negamax search algorithm as described in
        the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        def ab_negamax(game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if depth == 0 or len(game.get_legal_moves()) < 1:
                return None, self.score(game, game.active_player)

            best_move = None
            best_val = float("-inf")

            for a in game.get_legal_moves():
                _, v = ab_negamax(game.forecast_move(a), depth - 1, -beta, -max(alpha, best_val))
                current_score = -v

                if current_score > best_val:
                    best_move = a
                    best_val = current_score

                    if best_val >= beta:
                        return best_move, best_val
            return best_move, best_val

        best_move, _ = ab_negamax(game, depth, alpha, beta)

        return best_move


class AlphaBetaNegamaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening negamax
    search with alpha-beta pruning. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        depth_limit = self.search_depth

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            while True:
                best_move, best_val = self.alphabeta_negamax(game, depth_limit)
                depth_limit += 1

        except SearchTimeout:
            pass

        # print str(self)
        # print depth_limit

        return best_move

    def alphabeta_negamax(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implemented depth-limited negamax search algorithm with alpha-beta pruning.


        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        current_depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : boolean
            Keeping track of current player

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if len(game.get_legal_moves()) < 1 or depth == 0:
            return None, self.score(game, game.active_player)

        best_move = None
        best_val = float("-inf")

        moves = game.get_legal_moves()
        for a in moves:
            _, v = self.alphabeta_negamax(game.forecast_move(a), depth - 1, -beta,
                                          -alpha)
            v = -v

            if v > best_val:
                best_val = v
                best_move = a

            alpha = max(alpha, v)

            if alpha >= beta:
                return best_move, best_val

        return best_move, best_val


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        depth_limit = self.search_depth
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            while True:
                best_move = self.alphabeta(game, depth_limit)
                depth_limit += 1

        except SearchTimeout:
            return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):

        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        def max_value(game, alpha, beta, current_depth):

            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if current_depth == 0 or len(game.get_legal_moves()) < 1:
                return self.score(game, self)

            v = float("-inf")
            for move in game.get_legal_moves():
                v = max(v, min_value(game.forecast_move(move), alpha, beta, current_depth - 1))

                if v >= beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(game, alpha, beta, current_depth):

            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            if current_depth == 0 or len(game.get_legal_moves()) < 1:
                return self.score(game, self)

            v = float("inf")
            for move in game.get_legal_moves():
                v = min(v, max_value(game.forecast_move(move), alpha, beta, current_depth - 1))

                if v <= alpha:
                    return v
                beta = min(beta, v)
            return v

        best_move = None
        best_val = alpha

        for a in game.get_legal_moves():
            v = min_value(game.forecast_move(a), best_val, beta, depth - 1)

            if v > best_val:
                best_move = a
                best_val = v

        return best_move

    # def alphabeta_improved(self, game, depth, alpha=float("-inf"), beta=float("inf")):
    #
    #     """Implement depth-limited minimax search with alpha-beta pruning as
    #     described in the lectures.
    #
    #     This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
    #     https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
    #
    #     **********************************************************************
    #         You MAY add additional methods to this class, or define helper
    #              functions to implement the required functionality.
    #     **********************************************************************
    #
    #     Parameters
    #     ----------
    #     game : isolation.Board
    #         An instance of the Isolation game `Board` class representing the
    #         current game state
    #
    #     depth : int
    #         Depth is an integer representing the maximum number of plies to
    #         search in the game tree before aborting
    #
    #     alpha : float
    #         Alpha limits the lower bound of search on minimizing layers
    #
    #     beta : float
    #         Beta limits the upper bound of search on maximizing layers
    #
    #     Returns
    #     -------
    #     (int, int)
    #         The board coordinates of the best move found in the current search;
    #         (-1, -1) if there are no legal moves
    #
    #     Notes
    #     -----
    #         (1) You MUST use the `self.score()` method for board evaluation
    #             to pass the project tests; you cannot call any other evaluation
    #             function directly.
    #
    #         (2) If you use any helper functions (e.g., as shown in the AIMA
    #             pseudocode) then you must copy the timer check into the top of
    #             each helper function or else your agent will timeout during
    #             testing.
    #     """
    #
    #     if self.time_left() < self.TIMER_THRESHOLD:
    #         raise SearchTimeout()
    #
    #     def check_move(game):
    #
    #         board = game._board_state
    #
    #         configs = board[-3:]
    #         board = board[:-3]
    #
    #         board_matrix = np.array(board).reshape((7, 7))
    #
    #         board_0 = str(board + configs).__hash__()
    #
    #         if board_0 in self.boards:
    #             return self.boards[board_0]
    #         else:
    #             board_270 = str(np.append(np.rot90(board_matrix, k=1).flatten(), configs)).__hash__()
    #             if board_270 in self.boards:
    #                 return self.boards[board_270]
    #             else:
    #                 board_90 = str(np.append(np.rot90(board_matrix, k=-1).flatten(), configs)).__hash__()
    #
    #                 if board_90 in self.boards:
    #                     return self.boards[board_90]
    #                 else:
    #                     board_180 = str(np.append(np.rot90(board_matrix, 2).flatten(), configs)).__hash__()
    #                     if board_180 in self.boards:
    #                         return self.boards[board_180]
    #                     else:
    #                         return None, None
    #
    #     def max_value(game, alpha, beta, current_depth):
    #
    #         if self.time_left() < self.TIMER_THRESHOLD:
    #             raise SearchTimeout()
    #
    #         if current_depth == 0 or len(game.get_legal_moves()) < 1:
    #             return self.score(game, self)
    #
    #         v = float("-inf")
    #         for move in game.get_legal_moves():
    #
    #             updated_board = game.forecast_move(move)
    #             w, d = check_move(updated_board)
    #
    #             if w is None:
    #                 v = max(v, min_value(updated_board, alpha, beta, current_depth - 1))
    #                 key = updated_board.hash()
    #                 self.boards[key] = (v, depth)
    #             else:
    #                 if d >= current_depth:
    #                     print "bla1"
    #                     v = w
    #                 else:
    #                     v = max(v, min_value(updated_board, alpha, beta, current_depth - 1))
    #                     key = updated_board.hash()
    #                     self.boards[key] = (v, depth)
    #
    #             if v >= beta:
    #                 return v
    #             alpha = max(alpha, v)
    #         return v
    #
    #     def min_value(game, alpha, beta, current_depth):
    #
    #         if self.time_left() < self.TIMER_THRESHOLD:
    #             raise SearchTimeout()
    #
    #         if current_depth == 0 or len(game.get_legal_moves()) < 1:
    #             return self.score(game, self)
    #
    #         v = float("inf")
    #         for move in game.get_legal_moves():
    #
    #             updated_board = game.forecast_move(move)
    #             w, d = check_move(updated_board)
    #
    #             if w is None:
    #                 v = min(v, max_value(updated_board, alpha, beta, current_depth - 1))
    #                 key = updated_board.hash()
    #                 self.boards[key] = (v, depth)
    #             else:
    #                 if d >= current_depth:
    #                     print "bla2"
    #                     v = w
    #                 else:
    #                     v = min(v, max_value(updated_board, alpha, beta, current_depth - 1))
    #                     key = updated_board.hash()
    #                     self.boards[key] = (v, depth)
    #
    #             if v <= alpha:
    #                 return v
    #             beta = min(beta, v)
    #         return v
    #
    #     best_move = None
    #     best_val = alpha
    #
    #     for move in game.get_legal_moves():
    #
    #         updated_board = game.forecast_move(move)
    #         w, d = check_move(updated_board)
    #
    #         if w is None:
    #             v = min_value(updated_board, best_val, beta, depth - 1)
    #             key = updated_board.hash()
    #             self.boards[key] = (v, depth)
    #         else:
    #             if d >= depth:
    #                 print "bla3"
    #                 v = w
    #             else:
    #                 v = min_value(updated_board, best_val, beta, depth - 1)
    #                 key = updated_board.hash()
    #                 self.boards[key] = (v, depth)
    #
    #         if v > best_val:
    #             best_move = move
    #             best_val = v
    #
    #     return best_move


class AlphaBetaNegamaxTTPlayer(IsolationPlayer):

    EXACT = 2
    LOWER_BOUND = 1
    UPPER_BOUND = 0

    """Game-playing agent that chooses a move using iterative deepening negamax
    search with alpha-beta pruning. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        depth_limit = self.search_depth

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            while True:
                best_move, best_val = self.alphabeta_negamax(game, depth_limit)
                depth_limit += 1

        except SearchTimeout:
            pass

        # print str(self)
        # print depth_limit
        return best_move

    def alphabeta_negamax(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implemented depth-limited negamax search algorithm with alpha-beta pruning.


        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        current_depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : boolean
            Keeping track of current player

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TRANSPOSITION LOOK-UP
        # alpha_orig = alpha

        # key = game.hash()
        # if key in self.boards:
        #     entry = self.boards[key]
        #
        #     # >= vs >
        #     if entry[0] >= depth:
        #         flag = entry[1]
        #         value = entry[2]
        #         if flag == EXACT:
        #             return None, value
        #         elif flag == LOWER_BOUND:
        #             alpha = max(alpha, value)
        #         else:
        #             beta = min(beta, value)
        #
        #         if alpha >= beta:
        #             print "caracu"
        #             return None, value
        # if alpha >= beta:
        #     return None, value

        if len(game.get_legal_moves()) < 1 or depth == 0:
            return None, self.score(game, game.active_player)

        best_move = None
        best_val = float("-inf")

        alpha_orig = alpha
        key = game.hash()
        if key in self.boards:
            entry = self.boards[key]

            # >= vs >
            if entry[0] >= depth:
                flag = entry[1]
                value = entry[2]
                if flag == AlphaBetaNegamaxTTPlayer.EXACT:
                    return None, value
                elif flag == AlphaBetaNegamaxTTPlayer.LOWER_BOUND:

                    if value >= beta:
                        return None, value

                    if value > alpha:
                        alpha = value
                elif flag == AlphaBetaNegamaxTTPlayer.UPPER_BOUND:

                    if value <= alpha:
                        return None, value

                    if value < beta:
                        beta = value
            else:
                best_move = entry[3]



        # second_move = None
        # second_value = float("-inf")

        moves = game.get_legal_moves()

        # if best_move and second_move:
        #     moves = best_move + second_move + list(set(moves) - set(best_move + second_move))

        if best_move:
            moves = [best_move] + list(set(moves) - {best_move})

        for a in moves:
            _, v = self.alphabeta_negamax(game.forecast_move(a), depth - 1, -beta,
                                          -alpha)
            v = -v

            if v > best_val:

                # second_move = best_move
                # second_value = best_val

                best_val = v
                best_move = a

            if best_val >= beta:
                return best_move, best_val

            alpha = max(alpha, best_val)

            # alpha = max(alpha, v)
            #
            # if alpha >= beta:
            #     return best_move, best_val

        # TRANSPOSITION STORING
        if best_val <= alpha_orig:
            flag = AlphaBetaNegamaxTTPlayer.UPPER_BOUND
        elif best_val >= beta:
            flag = AlphaBetaNegamaxTTPlayer.LOWER_BOUND
        else:
            flag = AlphaBetaNegamaxTTPlayer.EXACT

        self.boards[game.hash()] = (depth, flag, best_val, best_move)

        return best_move, best_val
