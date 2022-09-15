from env.ultimate_ttt import UltimateTTT
from env.macros import *
class NegaMax:
    '''
    game: a game object
    target: the player whose result we want to seek (X or O)
    '''
    def __init__(self, state:dict) -> None:
        self.game = UltimateTTT(None, None, state)
    
    def run(self):
        '''
        return a (score, best move)
        score 1 denotes win for the current player
        score 0 denotes tie
        score -1 denotes loss for the current player
        '''
        # statically evaluate respect to the current player
        if self.game.outcome == X_WIN:
            return (1, None) if self.game.current_player == X else (-1, None)
        elif self.game.outcome == O_WIN:
            return (1, None) if self.game.current_player == O else (-1, None)
        elif self.game.outcome == TIE:
            return (0, None)
        else:
            valid_moves = self.game.next_valid_moves
            max_score = -2
            for move in valid_moves:
                self.game.update_state(move)
                # not interested in the best move for the opponent
                score, _ = self.run()
                score = -score  # negate the score for the current player
                self.game.undo()

                if score > max_score:
                    max_score = score
                    best_move = move

                # is a win postion already, prone the rest
                if max_score == 1:
                    return (1, best_move)

            return (max_score, best_move)
        