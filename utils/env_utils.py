from termcolor import colored
import numpy as np
from env.macros import *


def check_board(board: np.ndarray):
    '''
    board: np.ndarray -- a 3x3 board
    return the outcome of this 3x3 board (x win, o win, tie or incomplete)
    '''
    for i in range(3):
        # check row for x
        if all(board[i] == X):
            return X_WIN
        # check row for o
        elif all(board[i] == O):
            return O_WIN
        # check column for x
        elif all(board[:, i] == X):
            return X_WIN
        # check column for o
        elif all(board[:, i] == O):
            return O_WIN

    # check diagonal for x
    if board[0, 0] == board[1, 1] == board[2, 2] == X:
        return X_WIN
    elif board[0, 2] == board[1, 1] == board[2, 0] == X:
        return X_WIN
    # check diagonal for o
    if board[0, 0] == board[1, 1] == board[2, 2] == O:
        return O_WIN
    elif board[0, 2] == board[1, 1] == board[2, 0] == O:
        return O_WIN

    # No one has won at this point

    # check incompelte
    if any(board[0] == EMPTY) or any(board[1] == EMPTY) or any(board[2] == EMPTY):
        return INCOMPLETE
    # no slot left, must be tie
    else:
        return TIE


def get_valid_moves(inner_board: np.ndarray, outer_board: np.ndarray, previous_move: int):
    '''
    inner_board: np.ndarray -- the inner board of the game
    outer_board: np.ndarray -- the outer board corresponding to the inner board
    previous_move: int -- the ordinal number of the previous move
    return the valid moves for the next player as a tuple of ordinals sorted incrementally
    '''

    if previous_move is None:
        return tuple(range(81))

    assert previous_move in range(81), f'previous move not in [0,80]'
    assert inner_board.shape == (
        9, 9), f'inner board dimension {inner_board.shape} incompatible'
    assert outer_board.shape == (
        3, 3), f'outer board dimension {outer_board.shape} incompatible'

    valid_moves = []
    # get coordinate in the sub-board
    sub_row, sub_col = ordinal_to_coordinate(previous_move, 'sub')
    # if the outer board is incomplete, only the slots inside the sub-board are allowed
    if outer_board[sub_row, sub_col] == INCOMPLETE:
        inner_row_start, inner_col_start = sub_row*3, sub_col*3
        for row in range(inner_row_start, inner_row_start+3):
            for col in range(inner_col_start, inner_col_start+3):
                if inner_board[row, col] == EMPTY:
                    valid_moves.append(coordinate_to_ordinal((row, col)))
    # otherwise the next player can choose to play any valid sub-board
    else:
        # loop through outer board
        for outer_row in range(3):
            for outer_col in range(3):
                # can play on the current sub-board
                if outer_board[outer_row, outer_col] == INCOMPLETE:
                    inner_row_start, inner_col_start = outer_row*3, outer_col*3
                    for row in range(inner_row_start, inner_row_start+3):
                        for col in range(inner_col_start, inner_col_start+3):
                            if inner_board[row, col] == EMPTY:
                                valid_moves.append(
                                    coordinate_to_ordinal((row, col)))
    
    valid_moves.sort()
    return tuple(valid_moves)

def switch_player(player):
    assert player == X or player == O, f'player of value {player} not recognized'
    return O if player == X else X
# --------------------------------------------- transformation functionalities ---------------------------------------------

def ordinal_to_coordinate(ordinal: int, target_board: str = 'inner'):
    '''
    ordinal: int -- the ordinal position in [0,80]
    target_board: str -- the target board to map to, accepted options: "inner", "outer" and "sub"
    '''
    assert ordinal in range(81), f'ordinal {ordinal} not in range 0-80'

    row = ordinal // 9
    col = ordinal % 9
    if target_board == 'inner':
        return row, col
    elif target_board == 'outer':
        return row//3, col//3
    elif target_board == 'sub':
        return row % 3, col % 3
    else:
        raise ValueError(
            'target board not recognized, accepted options: "inner", "outer", "sub"')


def coordinate_to_ordinal(coordinate: tuple):
    '''
    coordinate: tuple(int, int) -- the coordinate to be mapped to ordinal (has to be coordinates of the inner board)
    maps inner board coordinates to ordinal number
    '''
    row, col = coordinate
    assert 0 <= row < 9, f'row index {row} not in range [0,8]'
    assert 0 <= col < 9, f'col index {col} not in range [0,8]'

    return row*9 + col


def inner_to_outer(inner_board: np.ndarray):
    '''
    inner_board: np.ndarray
    return the outer board derived from the input inner board
    '''
    assert inner_board.shape == (
        9, 9), f'illegal inner board dimension {inner_board.shape}'
    outer_board = np.zeros((3, 3), dtype=np.short)
    for outer_row in range(3):
        for outer_col in range(3):
            inner_row = outer_row*3
            inner_col = outer_col*3
            sub_board = inner_board[inner_row:inner_row +
                                    3, inner_col:inner_col+3]
            outer_board[outer_row, outer_col] = check_board(sub_board)
    return outer_board

# --------------------------------------------- displaying functionalities ---------------------------------------------


def print_line(color=False, length='long'):
    '''
    helper function to print a horizontal line
    color: bool -- swtich the color on and off
    length: str -- very long for game turn separation, long for inner board, short for outer board
    '''
    if length == 'long':
        print("  ", end='')
        if color:
            print(colored(' ---'*9, 'yellow', attrs=['bold']))
        else:
            print(' ---'*9)
    elif length == 'short':
        print("  ", end='')
        if color:
            print(colored(' ---'*3, 'yellow', attrs=['bold']))
        else:
            print(' ---'*3)
    elif length == 'very long':
        if color:
            print(colored('='*100, 'red', attrs=['bold']))
        else:
            print('-'*30)
    else:
        raise ValueError(
            'length not recognized, accepted lengths: "short", "long", "very long"')


def display_board(board: np.ndarray, mode: str = 'inner'):
    '''
    print the content of the board to terminal
    board: np.ndarray -- the board to be printed
    mode: str -- mode of one of 'inner', 'outer' and 'both'
    If the mode is set to 'inner' or 'both', then the input board must be an inner board.
    '''
    assert board.shape == (9, 9) or board.shape == (
        3, 3), f'illegal input board dimension {board.shape}'

    def display_inner(board):
        assert board.shape == (
            9, 9), f'illegal inner board dimension {board.shape}'
        # print col number
        indices = map(str, list(range(9)))
        indices = "   ".join(indices)
        indices = "    " + indices
        print(indices)
        for i, row in enumerate(board):
            # print horizontal line
            if i % 3 == 0:
                print_line(color=True)
            else:
                print_line(color=False)
            # print row number
            print(i, end=" ")
            # print markers and vertical lines
            for j, item in enumerate(row):
                print(colored('|', 'yellow', attrs=['bold']), end='') if j % 3 == 0 else print(
                    '|', end='')
                if item == X:
                    print(colored(' x ', 'cyan'), end='')
                elif item == O:
                    print(colored(' o ', 'red'), end='')
                else:
                    print('   ', end='')
            print(colored('|', 'yellow', attrs=['bold']))
        # print bottom line
        print_line(color=True)

    def display_outer(board):
        assert board.shape == (
            3, 3), f'illegal outer board dimension {board.shape}'
        # print the col numbers
        indices = map(str, list(range(3)))
        indices = "   ".join(indices)
        indices = "    " + indices
        print(indices)
        for i, row in enumerate(board):
            # print horizontal line
            print_line(color=True, length='short')
            # print the row numbers
            print(i, end=" ")
            # print markers and vertical lines
            for _, item in enumerate(row):
                print(colored('|', 'yellow', attrs=['bold']), end='')
                if item == X_WIN:
                    print(colored(' x ', 'cyan'), end='')
                elif item == O_WIN:
                    print(colored(' o ', 'red'), end='')
                elif item == TIE:
                    print(colored(' t ', 'grey'), end='')
                else:
                    print('   ', end='')
            print(colored('|', 'yellow', attrs=['bold']))
        # print bottom line
        print_line(color=True, length='short')

    if mode == 'inner':
        display_inner(board)
    elif mode == 'outer':
        if board.shape == (3, 3):
            display_outer(board)
        else:
            # derive the outer board
            outer_board = inner_to_outer(board)
            display_outer(outer_board)
    elif mode == 'both':
        display_inner(board)
        # derive the corresponding outer board
        outer_board = inner_to_outer(board)
        display_outer(outer_board)
    else:
        raise ValueError(
            f'mode {mode} not recognized, accepted modes: "inner", "outer", "both"')

def display_valid_moves(valid_moves: tuple):
    '''
    valid_moves: tuple -- the tuple of valid moves in ordinal form
    print the valid moves in coordinate format
    '''
    assert len(valid_moves) > 0
    valid_move_coord = map(ordinal_to_coordinate, valid_moves)
    valid_move_coord = map(str, list(valid_move_coord))
    valid_move_coord = ' '.join(valid_move_coord)
    print('valid moves:', valid_move_coord)

def main():
    board = np.zeros((9,9))
    board[0,0], board[1,1], board[2,2] = X, X, X
    board[0,3], board[0,4], board[0,5] = O, O, O
    board[0,6], board[1,7], board[1,8], board[2,6], board[2,7] = X, X, X, X, X
    board[0,7], board[0,8], board[1,6], board[2,8] = O, O, O, O
    display_board(board, 'both')

    valid_moves = get_valid_moves(board, inner_to_outer(board), 24)
    display_valid_moves(valid_moves)

if __name__ == '__main__':
    main()
