from env.ultimate_ttt import UltimateTTT
from env.macros import *
import random
class BooleanMinimax:
    def __init__(self, state:dict, bounded) -> None:
        self.game = UltimateTTT(None, None, state)
        self.root = state['current_player']
        self.bounded = bounded
    
    def boolean_or(self) -> bool:
        # statically evaluate
        if self.game.outcome == X_WIN:
            return (self.root==X), None
        elif self.game.outcome == O_WIN:
            return (self.root == O), None
        elif self.game.outcome == TIE:
            # tie is considered True if seeking bounded result, otherwise False
            return self.bounded, None
        else:
            legal_moves = self.game.next_valid_moves
            for move in legal_moves:
                self.game.update_state(move)
                result, _ = self.boolean_and()
                self.game.undo()
                if result:
                    return True, move # if one of them safisfies the condition, then it's True

            return False, random.choice(legal_moves)

    def boolean_and(self) -> bool:
        # statically evaluate
        if self.game.outcome == X_WIN:
            return (self.root == X), None
        elif self.game.outcome == O_WIN:
            return (self.root == O), None
        elif self.game.outcome == TIE:
            # tie is considered True if seeking bounded result, otherwise False
            return self.bounded, None
        else:
            legal_moves = self.game.next_valid_moves
            for move in legal_moves:
                self.game.update_state(move)
                result, _ = self.boolean_or()
                self.game.undo()
                if not result:
                    return False, move # if one of them does not satisfy the condition, then it's False

            return True, random.choice(legal_moves)


    def run(self):
        return self.boolean_or()
