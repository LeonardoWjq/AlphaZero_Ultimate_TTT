from players.player import Player
from solvers.negamax import NegaMax
from env.macros import *
class NegamaxPlayer(Player):
    def __init__(self, verbose=False) -> None:
        super().__init__()
        self.verbose = verbose
    def move(self, state: dict):
        player = state['current_player']
        solver = NegaMax(state)
        best_score, best_move = solver.run()
        if self.verbose:
            outcome_map = {1:'win', 0:'tie', -1:'loss'}
            player_map ={X:'X', O:'O'}
            print(f"Negamax says it's a {outcome_map[best_score]} for player {player_map[player]}.")
        return best_move