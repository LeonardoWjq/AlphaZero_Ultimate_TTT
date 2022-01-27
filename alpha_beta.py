from environment import UltimateTTT
from player import RandomPlayer
import time
class AlphaBeta:
    def __init__(self, player1, player2, root_player = 1, num_rand = 0) -> None:
        self.p1 = player1
        self.p2 = player2
        self.num_rand = num_rand
        self.game = UltimateTTT(self.p1, self.p2)
        self.root_player = root_player
    
    def random_play(self):
        for _ in range(self.num_rand):
            if self.game.game_end:
                break
            move = self.game.make_move()
            self.game.update_game_state(move)
    
    def alpha_beta_search(self, state, alpha, beta) -> int:
        # statically evaluate
        if state['game_end']:
            # if it is a tie
            if state['winner'] == 2:
                return 0
            # current player winning
            if state['winner'] == state['current']:
                return 1
            # current player losing
            else:
                return -1
        else:
            legal_moves = state['valid_move']
            for move in legal_moves:
                game = UltimateTTT(self.p1, self.p2, state)
                game.update_game_state(move)
                value = - self.alpha_beta_search(game.get_state(), -beta, -alpha)
                # update best to be alpha
                if (value > alpha):
                    alpha = value
                # beta cut
                if (value >= beta):
                    return beta
            
            return alpha
                


    def run(self):
        self.random_play()
        self.game.display_board()
        self.game.display_board(board='outer')
        

        start_time = time.time()
        # get the result for the current player
        value = self.alpha_beta_search(self.game.get_state(),float('-inf'), float('inf'))
        print(f'Time to run Alpha-Beta Search: {time.time() - start_time}')

        return value

    
def main():
    player1 = RandomPlayer()
    player2 = RandomPlayer()
    agent = AlphaBeta(player1, player2, num_rand=50)
    result = agent.run()
    current_player = 'x' if agent.game.current_player == 1 else 'o'
    print(f'Value of the current player {current_player} is {result}.')

if __name__ == '__main__':
    main()