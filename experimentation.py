from environment import UltimateTTT
from negamax import Negamax
from boolean_minimax import BooleanMinimax
from alpha_beta import AlphaBeta
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
    random_play(game,53,1)

    negamax_agent = Negamax(game)
    boolean_minimax_agent = BooleanMinimax(game)
    alpha_beta_agent = AlphaBeta(game)
    
    print(negamax_agent.run())
    print(boolean_minimax_agent.run())
    print(alpha_beta_agent.run())

if __name__ == '__main__':
    main()
