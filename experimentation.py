from environment import UltimateTTT
from negamax import Negamax
from boolean_minimax import BooleanMinimax
from alpha_beta import AlphaBeta
from alpha_beta_tt import Alpha_Beta_TT
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
            alpha_beta_tt_agent = Alpha_Beta_TT(game)
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

def test_MCTS(base_sim_num = 20, start_sim_num = 10, end_sim_num = 100, interval = 10, game_num = 20, seed = 0):
    # set random seed for reproducibility
    np.random.seed(seed)
    # uniform random prior policy
    pol = RandomPolicy()
    # default(uniform random) simulation policy
    sim = RandomPlayer()
    # baseline player
    base_player = MCTSPlayer(pol,sim,base_sim_num)

    win_rate = []
    lose_rate = []
    draw_rate = []

    for test_sim_num in tqdm(range(start_sim_num, end_sim_num+1, interval)):

        test_player = MCTSPlayer(pol,sim,test_sim_num)
        num_wins = 0
        num_draws = 0
        num_loses = 0

        for game in range(game_num):
            # test player plays first
            if game % 2 == 0:
                game = UltimateTTT(test_player,base_player)
                game.play()
                if game.winner == 1:
                    num_wins += 1
                elif game.winner == -1:
                    num_loses += 1
                else:
                    num_draws += 1
            # base player plays first
            else: 
                game = UltimateTTT(base_player,test_player)
                game.play()
                if game.winner == 1:
                    num_loses += 1
                elif game.winner == -1:
                    num_wins += 1
                else:
                    num_draws += 1

            base_player.reset()
            test_player.reset()
        
        win_rate.append((num_wins/game_num)*100)
        lose_rate.append((num_loses/game_num)*100)
        draw_rate.append((num_draws/game_num)*100)

    record = [win_rate, lose_rate, draw_rate]
    with open('test_mcts.txt','wb') as out_file:
        pickle.dump(record,out_file)


def plot_run_time(start_rand_num = 52, end_rand_num = 62, interval = 1):
    # read the data
    with open('run_time.txt','rb') as in_file:
        record = pickle.load(in_file)
    names = ['Boolean Minimax', 'Negamax', 'Alpha-Beta', 'Alpha-Beta TT', 'Proof Number Search']
    record = np.array(record)
    mean = np.mean(record,axis = 0)
    plt.figure()
    for i in range(5):
        plt.plot(range(start_rand_num, end_rand_num+1, interval),np.log(mean[:,i]),label=names[i])
    
    plt.legend()
    plt.grid()
    plt.title('Solver Running Time Average on log Scale against Number of Initial Random Plays')
    plt.xlabel('Number of Initial Random Plays')
    plt.ylabel('Natural log of Running Time')
    plt.show()

def plot_test_mcts(start_sim_num = 10, end_sim_num = 100, interval = 10, game_num=20):
    with open('test_mcts.txt','rb') as in_file:
        record = pickle.load(in_file)
    
    record = np.array(record)
    plt.plot(range(start_sim_num,end_sim_num+1,interval),record[0,:],label='Win')
    plt.plot(range(start_sim_num,end_sim_num+1,interval),record[1,:],label='Lose')
    plt.plot(range(start_sim_num,end_sim_num+1,interval),record[2,:],label='Draw')
    plt.title(f'Performance of the MCTS Player with Varying Degrees of Simulation againt the Baseline Player')
    plt.xlabel('Number of Simulations')
    plt.ylabel(f'Percentage on {game_num} Games')
    plt.legend()
    plt.grid()
    plt.show()
    
def main():
    player = RandomPlayer()
    sample_game = UltimateTTT(player,player)
    random_play(sample_game)
    sample_game.display_board()
    sample_game.display_board('outer')
    plot_run_time()
    plot_test_mcts()
    

if __name__ == '__main__':
    main()
