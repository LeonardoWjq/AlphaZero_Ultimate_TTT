from players.player import Player
from solvers.alpha_beta import AlphaBeta
from env.macros import *

class AlphaBetaPlayer(Player):
    def __init__(self, verbose=False) -> None:
        super().__init__()
        self.verbose = verbose

    def move(self, state: dict):
        player = state['current_player']
        solver = AlphaBeta(state)
        best_score, best_move = solver.run(-1, 1) # we know the score is bounded by [-1, 1]
        if self.verbose:
            outcome_map = {1:'win', 0:'tie', -1:'loss'}
            player_map ={X:'X', O:'O'}
            print(f"Alpha-beta says it's a {outcome_map[best_score]} for player {player_map[player]}.")
        return best_move