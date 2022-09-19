from env.macros import *
from env.ultimate_ttt import UltimateTTT
from players.mcts_player import MCTSPlayer
from players.human_player import HumanPlayer
from utils.test_utils import generate_random_game
from tqdm import tqdm

def test_mcts_player(rollout_num=0, seed=0):
    random_state = generate_random_game(rollout_num, seed)
    px = HumanPlayer()
    po = MCTSPlayer(verbose=True)
    game = UltimateTTT(px, po, random_state)
    game.play(display='all')

def self_play(num_games = 20, sim_1 = 200, sim_2 = 200, c1 = 1.4, c2 = 1.4):
    player1 = MCTSPlayer(num_simulation=sim_1, explore_factor=c1)
    player2 = MCTSPlayer(num_simulation=sim_2, explore_factor=c2)

    player1_win = 0
    player2_win = 0
    tie = 0
    for i in tqdm(range(num_games)):
        game = UltimateTTT(player1, player2) if i%2 == 0 else UltimateTTT(player2, player1)
        game.play()
        if game.outcome == X_WIN:
            if i%2 == 0: player1_win += 1
            else: player2_win += 1
        elif game.outcome == O_WIN:
            if i%2 == 0: player2_win += 1
            else: player1_win += 1
        elif game.outcome == TIE:
            tie += 1
        
        player1.reset()
        player2.reset()
    
    print(f'player1 win: {player1_win}')
    print(f'player2 win: {player2_win}')
    print(f'tie: {tie}')

    return player1_win, player2_win, tie




if __name__ == '__main__':
    self_play(sim_1=30,sim_2=10)