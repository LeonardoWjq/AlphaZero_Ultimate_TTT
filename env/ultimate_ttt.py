import numpy as np
from termcolor import colored
from macros import *
from utils.env_utils import *
from players.random_player import RandomPlayer
from players.human_player import HumanPlayer
from collections import namedtuple

Step = namedtuple('Step', ['previous_move', 'move'])


class UltimateTTT:
    def __init__(self, player_x, player_o, state=None) -> None:
        if state:
            self.inner_board = np.copy(state['inner_board'])
            self.outer_board = inner_to_outer(self.inner_board)
            self.current_player = state['current_player']
            self.outcome = state['outcome']
            self.previous_move = state['previous_move']
            self.next_valid_moves = get_valid_moves(
                self.inner_board, self.outer_board, self.previous_move)
            self.history = state['history'].copy()
        else:
            self.inner_board = np.zeros((9, 9), dtype=np.short)
            self.outer_board = np.zeros((3, 3), dtype=np.short)
            self.current_player = X
            self.outcome = INCOMPLETE
            self.previous_move = None
            self.next_valid_moves = tuple(range(0, 81))
            self.history = []

        self.player_x = player_x
        self.player_o = player_o

    def play(self, display: str = 'none'):
        '''
        display: str -- the option to control the level of display:
                        all: print the board and the moves at every step
                        end: only print the game outcome
                        none: do not print anything
        play the game until the end (X wins, O wins or Tie)
        '''
        assert display in (
            'all', 'end', 'none'), f'display option {display} invalid, accepted arguments: "all", "end", "none"'
        while self.outcome == INCOMPLETE:
            if display == 'all':
                # print the board
                print("The inner board:\n")
                display_board(self.inner_board, 'inner')
                print()
                print("The outer board:\n")
                display_board(self.outer_board, 'outer')
                print()
            player = 'x' if self.current_player == X else 'o'
            if display == 'all':
                print(f"Player {player}'s turn to make a move:\n")
                display_valid_moves(self.next_valid_moves)

            # make a move
            move = self.make_move()
            coordinate = ordinal_to_coordinate(move)

            if display == 'all':
                # print a line of separation
                print_line(True, 'very long')
                print(f'Player {player} played position {coordinate}.\n')

            # update game state
            self.update_state(move)

        if display == 'all' or display == 'end':
            print(colored("Game over!\n", 'green', attrs=['bold']))
            # display result
            if self.outcome == X_WIN:
                print(colored('The winner is player x!\n',
                      'green', attrs=['bold']))
            elif self.outcome == O_WIN:
                print(colored('The winner is player o!\n',
                      'green', attrs=['bold']))
            else:
                print(colored('The game is a tie.\n',
                      'green', attrs=['bold']))

            # print the board
            print("The inner board:\n")
            display_board(self.inner_board)
            print()
            print("The outer board:\n")
            display_board(self.outer_board, 'outer')

    def get_state(self):
        '''
        return the current game state as a dict
        '''
        state = {
            "inner_board": np.copy(self.inner_board),
            "current_player": self.current_player,
            "outcome": self.outcome,
            "previous_move": self.previous_move,
            "history": self.history.copy()
        }
        return state

    def make_move(self):
        '''
        return the move in ordinal form selected by the current player
        '''
        current_state = self.get_state()
        if self.current_player == X:
            candidate_move = self.player_x.move(current_state)
            assert candidate_move in self.next_valid_moves, f'move made by player X is not valid'
        else:
            candidate_move = self.player_o.move(current_state)
            assert candidate_move in self.next_valid_moves, f'move made by player O is not valid'
        return candidate_move

    def update_state(self, move: int):
        '''
        move: int -- the ordinal format of a move
        update the game state after making the move
        '''
        self.history.append(Step(self.previous_move, move))
        self.update_board(move)
        self.update_outcome()
        self.next_valid_moves = get_valid_moves(
            self.inner_board, self.outer_board, move)
        self.previous_move = move
        self.current_player = switch_player(self.current_player)

    def undo(self):
        '''
        undo the previous move and restore the game state
        '''
        try:
            previous_move, move = self.history.pop()
        except IndexError:
            print(colored('no history left in the stack, undo unsuccessful', 'red'))
            return

        self.undo_board(move)
        self.update_outcome()
        self.next_valid_moves = get_valid_moves(
            self.inner_board, self.outer_board, previous_move)
        self.previous_move = previous_move
        self.current_player = switch_player(self.current_player)

    def update_board(self, move):
        '''
        move: int -- the ordinal form of the move
        update the inner board and the outer board
        after playing the move
        '''
        # update inner board
        inner_coord = ordinal_to_coordinate(move)
        self.inner_board[inner_coord] = self.current_player
        # update the outer board
        outer_row, outer_col = ordinal_to_coordinate(
            move, target_board='outer')
        sub_board = self.inner_board[outer_row*3:outer_row*3+3,
                                     outer_col*3:outer_col*3+3]
        self.outer_board[outer_row, outer_col] = check_board(sub_board)

    def undo_board(self, move):
        '''
        move: int -- the ordinal form of the move
        update the inner board and the outer board
        after undoing the move
        '''
        inner_coord = ordinal_to_coordinate(move)
        self.inner_board[inner_coord] = EMPTY  # undo inner board position
        outer_row, outer_col = ordinal_to_coordinate(
            move, target_board='outer')
        sub_board = self.inner_board[outer_row*3:outer_row*3+3,
                                     outer_col*3:outer_col*3+3]
        self.outer_board[outer_row, outer_col] = check_board(
            sub_board)  # update outer board

    def update_outcome(self):
        '''
        update the outcome of the game
        '''
        self.outcome = check_board(self.outer_board)

    def switch(self):
        '''
        switch the current player
        '''
        self.current_player = switch_player(self.current_player)

    def display_state(self):
        '''
        print the current game state
        '''
        print('inner board:')
        display_board(self.inner_board)
        print('outer board:')
        display_board(self.outer_board, 'outer')

        outcome_map = {X_WIN: 'x win', O_WIN: 'o win',
                       TIE: 'tie', INCOMPLETE: 'incomplete'}
        print('outcome of the game:', outcome_map[self.outcome])

        player_map = {X: 'x', O: 'o'}
        print('player:', player_map[self.current_player])

        if self.previous_move is None:
            print('previous move does not exist')
        else:
            print('previous move:', ordinal_to_coordinate(self.previous_move))

        display_valid_moves(self.next_valid_moves)


def main():
    px = RandomPlayer()
    po = HumanPlayer()
    game = UltimateTTT(px, po)
    game.play('all')


if __name__ == '__main__':
    main()
