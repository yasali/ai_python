"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # Count X's and O's
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    
    # If X has more moves, it's O's turn, otherwise it's X's turn
    return O if x_count > o_count else X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))
    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Make a deep copy of the board
    new_board = [row[:] for row in board]
    i, j = action
    
    # Check if the action is valid
    if i < 0 or i > 2 or j < 0 or j > 2 or board[i][j] != EMPTY:
        raise Exception("Invalid action")
    
    # Make the move
    new_board[i][j] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check rows
    for row in board:
        if row.count(X) == 3:
            return X
        if row.count(O) == 3:
            return O
    
    # Check columns
    for j in range(3):
        if board[0][j] == board[1][j] == board[2][j] != EMPTY:
            return board[0][j]
    
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]
    
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Check if there's a winner
    if winner(board) is not None:
        return True
    
    # Check if the board is full
    return all(cell != EMPTY for row in board for cell in row)


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    winner_player = winner(board)
    if winner_player == X:
        return 1
    elif winner_player == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    
    def max_value(board, alpha, beta):
        if terminal(board):
            return utility(board)
        v = float('-inf')
        for action in actions(board):
            v = max(v, min_value(result(board, action), alpha, beta))
            alpha = max(alpha, v)
            if beta <= alpha:
                break
        return v
    
    def min_value(board, alpha, beta):
        if terminal(board):
            return utility(board)
        v = float('inf')
        for action in actions(board):
            v = min(v, max_value(result(board, action), alpha, beta))
            beta = min(beta, v)
            if beta <= alpha:
                break
        return v
    
    current_player = player(board)
    best_action = None
    
    if current_player == X:
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in actions(board):
            value = min_value(result(board, action), alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)
    else:
        best_value = float('inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in actions(board):
            value = max_value(result(board, action), alpha, beta)
            if value < best_value:
                best_value = value
                best_action = action
            beta = min(beta, best_value)
    
    return best_action
