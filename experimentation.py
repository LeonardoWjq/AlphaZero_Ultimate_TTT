from unicodedata import name
from environment import UltimateTTT
from negamax import Negamax
from player import RandomPlayer
import numpy as np

def random_play(game:UltimateTTT, num_play = 50, seed = 0):
    np.random.seed(seed)
    for _ in range(num_play):
        if game.game_end:
            break
        move = game.make_move()
        game.update_game_state(move)


def main():
    player = RandomPlayer()
    game = UltimateTTT(player,player)
    random_play(game,50,2)
    agent = Negamax(game)
    res = agent.run(True)
    print(res)

if __name__ == '__main__':
    main()
