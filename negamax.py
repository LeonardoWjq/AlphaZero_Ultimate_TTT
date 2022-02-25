from environment import UltimateTTT
import time
class Negamax:
    '''
    game: a game object
    target_player: the player whose result we want to seek
                   1 for x and -1 for o
    '''
    def __init__(self, game:UltimateTTT, target_player = 1) -> None:
        self.game = game
        self.root = game.current_player
        self.target = target_player
    
    '''
    The negamax algorithm
    Return a bound: True -> win or draw; False -> lose or draw
    '''
    def negamax(self) -> bool:
        # statically evaluate
        if self.game.game_end:
            return self.game.winner == self.game.current_player
        else:
            legal_moves = self.game.next_valid_move
            for move in legal_moves:
                # game = UltimateTTT(self.p1, self.p2, state)
                self.game.update_game_state(move)
                oppo_win = self.negamax()
                self.game.undo()
                if not oppo_win:
                    # draw or win
                    return True

            # draw or lose 
            return False

    '''
    display: whether or not to display the board to begin with
    This method runs the negamax algorithm and records the time
    '''
    def run(self, display = False):
        if display:
            self.game.display_board()
            self.game.display_board(board='outer')

        start_time = time.time()
        # get the result for the current player
        res = self.negamax()
        time_used = time.time() - start_time

        print(f'Time to run Negamax: {time_used}')

        # root player is the same as the target player
        if self.root == self.target:
            return res, time_used
        # root player is different from the target player
        else:
            return (not res), time_used