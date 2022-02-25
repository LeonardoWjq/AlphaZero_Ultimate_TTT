from environment import UltimateTTT
from negamax import Negamax
from boolean_minimax import BooleanMinimax
from alpha_beta import AlphaBeta
from transposition_table import Transposition
from pns import PNS
from player import RandomPlayer,MCTSPlayer
from policy import RandomPolicy
from collections import namedtuple
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt

RunningTime = namedtuple('RunningTime', ['bool_minimax', 'negamax', 'alpha_beta', 'alpha_beta_tt', 'pns'])

def random_play(game:UltimateTTT, num_play = 50, seed = 0):
    np.random.seed(seed)
    for _ in range(num_play):
        if game.game_end:
            break
        move = game.make_move()
        game.update_game_state(move)

def test_runtime(start_rand_num = 52, end_rand_num = 62, interval = 1, num_trials = 10):
    record_by_trial = []
    player = RandomPlayer()
    for trial in tqdm(range(num_trials)):
        record_by_num_rand = []
        for num_rand in range(start_rand_num, end_rand_num+1, interval):
            game = UltimateTTT(player,player)
            random_play(game,num_rand,trial+1)

            boolean_minimax_agent = BooleanMinimax(game)
            negamax_agent = Negamax(game)   
            alpha_beta_agent = AlphaBeta(game)
            alpha_beta_tt_agent = Transposition(game)
            pns_agent = PNS(game)

            _,bool_minimax_time = boolean_minimax_agent.run()
            _,negamax_time = negamax_agent.run()
            _,alpha_beta_time = alpha_beta_agent.run()
            _,alpha_beta_tt_time = alpha_beta_tt_agent.run()
            _,pns_time = pns_agent.run()

            record_by_agent = RunningTime(bool_minimax=bool_minimax_time,
                                          negamax=negamax_time,
                                          alpha_beta=alpha_beta_time,
                                          alpha_beta_tt=alpha_beta_tt_time,
                                          pns=pns_time)
            record_by_num_rand.append(record_by_agent)
        
        record_by_trial.append(record_by_num_rand)

    with open('run_time.txt','wb') as file_out:
        pickle.dump(record_by_trial,file_out)

def test_MCTS():
    pol = RandomPolicy()
    sim = RandomPlayer()
    rand_player = RandomPlayer()
    mcts_player = MCTSPlayer(pol,sim,50)
    mcts2_player = MCTSPlayer(pol,sim,20) 
    wins = 0
    draws = 0
    loses = 0
    for i in tqdm(range(10)):
        game = UltimateTTT(mcts_player,mcts2_player)
        game.play()
        if game.winner == 1:
            wins += 1
        elif game.winner == -1:
            loses += 1
        else:
            draws += 1
        mcts_player.reset()
        mcts2_player.reset()

    
    print('wins',wins)
    print('loses',loses)
    print('draws',draws)


def plot_run_time(start_rand_num = 52, end_rand_num = 62, interval = 1, num_trials = 10):
    # read the data
    with open('run_time.txt','rb') as in_file:
        record = pickle.load(in_file)
    names = ['Boolean Minimax', 'Negamax', 'Alpha-Beta', 'Alpha-Beta TT', 'Proof Number Search']
    record = np.array(record)
    mean = np.mean(record,axis = 0)
    for i in range(5):
        plt.plot(range(start_rand_num, end_rand_num+1, interval),np.log(mean[:,i]),label=names[i])
    
    plt.legend()
    plt.grid()
    plt.title('Solver Running Times on log Scale against Number of Initial Random Plays')
    plt.xlabel('Number of Initial Random Plays')
    plt.ylabel('Natural log of Running Time')
    plt.show()


def main():
    test_MCTS()

if __name__ == '__main__':
    main()
