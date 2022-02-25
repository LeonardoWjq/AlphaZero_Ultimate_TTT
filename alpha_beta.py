from environment import UltimateTTT
from player import RandomPlayer
import time
class AlphaBeta:
    def __init__(self, game:UltimateTTT, target_player = 1) -> None:
        self.game = game
        self.root = game.current_player
        self.target = target_player
    
    def alpha_beta_search(self,alpha, beta) -> int:
        # statically evaluate
        if self.game.game_end:
            # if it is a tie
            if self.game.winner == 2:
                return 0
            # current player winning
            if self.game.winner == self.game.current_player:
                return 1
            # current player losing
            else:
                return -1
        else:
            legal_moves = self.game.next_valid_move
            for move in legal_moves:
                self.game.update_game_state(move)
                value = - self.alpha_beta_search(-beta, -alpha)
                self.game.undo()
                # update best to be alpha
                if (value > alpha):
                    alpha = value
                # beta cut
                if (value >= beta):
                    return beta
            
            return alpha
                


    def run(self, display = False):
        if display:
            self.game.display_board()
            self.game.display_board(board='outer')
        
        start_time = time.time()
        # get the result for the current player
        value = self.alpha_beta_search(float('-inf'), float('inf'))
        time_used = time.time() - start_time
        print(f'Time to run Alpha-Beta Search: {time_used}')
        res = None
        if value == 0:
            res = 0
        elif value == 1:
            res =  1 if self.root == self.target else -1
        else:
            res = -1 if self.root == self.target else 1

        return res, time_used

    
# def main():
#     player1 = RandomPlayer()
#     player2 = RandomPlayer()
#     agent = AlphaBeta(player1, player2, num_rand=50)
#     result = agent.run()
#     current_player = 'x' if agent.game.current_player == 1 else 'o'
#     print(f'Value of the current player {current_player} is {result}.')

# if __name__ == '__main__':
#     main()