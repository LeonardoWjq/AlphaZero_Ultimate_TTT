import random

from utils.env_utils import get_valid_moves, inner_to_outer

from players.player import Player


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def move(self, state: dict):
        inner_board = state['inner_board']
        outer_board = inner_to_outer(inner_board)
        prev_move = state['previous_move']
        valid_moves = get_valid_moves(inner_board, outer_board, prev_move)
        return random.choice(valid_moves)
