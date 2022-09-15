from env.ultimate_ttt import UltimateTTT
from env.macros import *


class AlphaBeta:
    def __init__(self, state: dict) -> None:
        self.game = UltimateTTT(None, None, state)

    def run(self, alpha, beta) -> int:
        # statically evaluate
        if self.game.outcome == X_WIN:
            return (1, None) if self.game.current_player == X else (-1, None)
        elif self.game.outcome == O_WIN:
            return (1, None) if self.game.current_player == O else (-1, None)
        elif self.game.outcome == TIE:
            return (0, None)
        else:
            valid_moves = self.game.next_valid_moves
            for move in valid_moves:
                self.game.update_state(move)
                score, _ = self.run(-beta, -alpha)
                score = -score
                self.game.undo()

                # update alpha
                if (score > alpha):
                    alpha = score
                # beta cut
                if (score >= beta):
                    return (beta, move)

            return (alpha, move)