from environment import UltimateTTT
from player import RandomPlayer
import time
class BooleanMinimax:
    def __init__(self, game:UltimateTTT , target_player = 1) -> None:
        self.game = game
        self.root = game.current_player
        self.target = target_player
    
    def boolean_or(self) -> bool:
        # statically evaluate
        if self.game.game_end:
            return self.game.winner == self.target
        else:
            legal_moves = self.game.next_valid_move
            for move in legal_moves:
                self.game.update_game_state(move)
                winning = self.boolean_and()
                self.game.undo()
                # player wins
                if winning:
                    return True

            # player draws or loses
            return False



    def boolean_and(self) -> bool:
        # statically evaluate
        if self.game.game_end:
            return self.game.winner == self.target
        else:
            legal_moves = self.game.next_valid_move
            for move in legal_moves:
                self.game.update_game_state(move)
                winning = self.boolean_or()
                self.game.undo()
                # player loses or draws
                if not winning:
                    return False

            # player wins
            return True


    def run(self, display = False):
        if display:
            self.game.display_board()
            self.game.display_board(board='outer')
        start_time = time.time()
        # if it is the target player to move use boolean or
        if self.root == self.target:
            res = self.boolean_or()
        else:
            # use boolean and otherwise
            res = self.boolean_and()
        
        time_used = time.time() - start_time

        print(f'Time to run Boolean Minimax: {time_used}')

        return res, time_used