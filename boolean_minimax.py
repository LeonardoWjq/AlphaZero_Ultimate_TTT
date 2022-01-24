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
    
    def boolean_or(self, state) -> bool:
        # statically evaluate
        if state['game_end']:
            return state['winner'] == self.root_player
        else:
            legal_moves = state['valid_move']
            for move in legal_moves:
                game = UltimateTTT(self.p1, self.p2, state)
                game.update_game_state(move)
                # if one of them is True, return True
                if self.boolean_and(game.get_state()):
                    return True
            # not a single winning move
            return False



    def boolean_and(self, state) -> bool:
        # statically evaluate
        if state['game_end']:
            return state['winner'] == self.root_player
        else:
            legal_moves = state['valid_move']
            for move in legal_moves:
                game = UltimateTTT(self.p1, self.p2, state)
                game.update_game_state(move)
                # is one of them is not winning, return False
                if not self.boolean_or(game.get_state()):
                    return False

            # all children are winning
            return True


    def run(self):
        self.random_play()
        self.game.display_board()
        self.game.display_board(board='outer')
        start_time = time.time()
        # if it is the root player to move use boolean or
        if self.game.current_player == self.root_player:
            res = self.boolean_or(self.game.get_state())
        else:
            # use boolean and otherwise
            res = self.boolean_and(self.game.get_state())

        print(f'Time to run Boolean Minimax: {time.time() - start_time}')

        return res

    
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