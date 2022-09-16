from players.player import Player
from solvers.pns import PNS
from env.macros import *

class PNSPlayer(Player):
    def __init__(self, verbose=False) -> None:
        super().__init__()
        self.verbose = verbose

    def move(self, state: dict):
        player = state['current_player']
        player_map ={X:'X', O:'O'}

        bounded_solver = PNS(state, bounded=True)
        exact_solver = PNS(state, bounded=False)

        bounded_res, bounded_move = bounded_solver.run()
        if not bounded_res: # root player is losing
            if self.verbose:
                print(f"Proof Number Search says it's a loss for player {player_map[player]}")
            return bounded_move
        else: # root player is at least drawing
            exact_res, exact_move = exact_solver.run()
            if exact_res: # root player is winning
                if self.verbose:
                    print(f"Proof Number Search says it's a win for player {player_map[player]}")
                return exact_move
            else: # root player is tying
                if self.verbose:
                    print(f"Proof Number Search says it's a tie for player {player_map[player]}")
                return bounded_move