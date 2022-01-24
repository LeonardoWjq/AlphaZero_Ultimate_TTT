from environment import UltimateTTT
from player import RandomPlayer
import time
class BooleanMinimax:
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
    
    def negamax(self, state) -> bool:
        # statically evaluate
        if state['game_end']:
            return state['winner'] == state['current']
        else:
            legal_moves = state['valid_move']
            for move in legal_moves:
                game = UltimateTTT(self.p1, self.p2, state)
                game.update_game_state(move)
                if not self.negamax(game.get_state()):
                    return True
            # not a single winning move
            return False


    def run(self):
        self.random_play()
        self.game.display_board()
        self.game.display_board(board='outer')
        current_player = self.game.current_player

        start_time = time.time()
        # get the result for the current player
        res = self.negamax(self.game.get_state())
        print(f'Time to run Boolean Minimax: {time.time() - start_time}')

        if current_player == self.root_player:
            return res
        else:
            return not res

    
def main():
    player1 = RandomPlayer()
    player2 = RandomPlayer()
    agent = BooleanMinimax(player1, player2, num_rand=50)
    result = agent.run()
    if result:
        print('Root Player is Winning.')
    else:
        print('Root Player is Not Winning.')

if __name__ == '__main__':
    main()