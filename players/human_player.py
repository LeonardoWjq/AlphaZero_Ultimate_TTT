from players.player import Player
from utils.env_utils import inner_to_outer, get_valid_moves, coordinate_to_ordinal, display_valid_moves
class HumanPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
    def move(self, state:dict)->int:
        inner_board = state['inner_board']
        previous_move = state['previous_move']
        outer_board = inner_to_outer(inner_board)
        valid_moves = get_valid_moves(inner_board, outer_board, previous_move)

        valid = False
        while not valid:
            usr_input = input('Please enter a position in row,column format: ')
            try:
                row, col = usr_input.split(',')
                row, col = int(row), int(col)
                ordinal_move = coordinate_to_ordinal((row, col))
                assert ordinal_move in valid_moves
                valid = True
            except ValueError:
                print('Illegal input. Expect row and column to be integers in [0,8]')
            except AssertionError:
                print('The selected move is not valid.')
                display_valid_moves(valid_moves)
            
        
        return ordinal_move