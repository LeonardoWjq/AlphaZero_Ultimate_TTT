from player import MCTSPNSPlayer, MCTSPlayer
from environment import UltimateTTT
from tqdm import tqdm
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

def run_trials(simulations = [100, 200, 300], rand_seed = 1):
    outcome = []
    np.random.seed(rand_seed)
    for sim_num in simulations:
        mctspns_agt = MCTSPNSPlayer(None, None, sim_num)
        mcts_agt = MCTSPlayer(None, None, sim_num)
        win, draw, loss = 0, 0, 0
        infos = []
        for j in tqdm(range(20)):
            if j % 2 == 0:
                game = UltimateTTT(mctspns_agt, mcts_agt, keep_history=False)
                game.play()
                if game.winner == 1:
                    win += 1
                elif game.winner == -1:
                    loss += 1
                else:
                    draw += 1
            else:
                game = UltimateTTT(mcts_agt, mctspns_agt, keep_history=False)
                game.play()
                if game.winner == 1:
                    loss += 1
                elif game.winner == -1:
                    win += 1
                else:
                    draw += 1
            
            infos.append(mctspns_agt.get_info())
            mctspns_agt.reset()
            mcts_agt.reset()
        
        play_result = {'win':win, 'draw':draw, 'loss':loss, 'info':infos}
        outcome.append((sim_num, play_result))

        with open('mctspns_outcome.pickle','wb') as fp:
            pkl.dump(outcome, fp)

def plot_figure():
    '''
    draw MCTS-PNS vs MCTS outcome
    '''
    with open('mctspns_outcome.pickle','rb') as fp:
        mctspns = pkl.load(fp)
    
    sim_nums = []
    wins = []
    draws = []
    losses = []
    for num, record in mctspns:
        sim_nums.append(str(num))
        wins.append(record['win'])
        draws.append(record['draw'])
        losses.append(record['loss'])
    
    wins = np.array(wins)
    draws = np.array(draws)
    losses = np.array(losses)

    width = 0.35
    fig, ax = plt.subplots()
    ax.set_yticks(np.arange(0,21,2))
    ax.bar(sim_nums, losses, width, label='Loss')
    ax.bar(sim_nums, draws, width, bottom=losses, label='Draw')
    ax.bar(sim_nums, wins, width, bottom= losses + draws, label = 'Win')
    ax.set_ylabel('Outcomes')
    ax.set_xlabel('# of Simulations per Move')
    ax.set_title('Outcomes of 20 Games with 100, 200 and 300 Simulations per Move')
    ax.legend()
    plt.savefig('MCTSPNS.png')


    '''
    draw MCTS vs MCTS outcome
    '''
    with open('mcts_outcome.pickle','rb') as fp:
        mcts = pkl.load(fp)
    
    sim_nums = []
    wins = []
    draws = []
    losses = []
    for num, record in mcts:
        sim_nums.append(str(num))
        wins.append(record['win'])
        draws.append(record['draw'])
        losses.append(record['loss'])
    
    wins = np.array(wins)
    draws = np.array(draws)
    losses = np.array(losses)

    width = 0.35
    fig, ax = plt.subplots()
    ax.set_yticks(np.arange(0,21,2))
    ax.bar(sim_nums, losses, width, label='Loss')
    ax.bar(sim_nums, draws, width, bottom=losses, label='Draw')
    ax.bar(sim_nums, wins, width, bottom= losses + draws, label = 'Win')
    ax.set_ylabel('Outcomes')
    ax.set_xlabel('# of Simulations per Move')
    ax.set_title('Outcomes of 20 Games with 100, 200 and 300 Simulations per Move')
    ax.legend()
    plt.savefig('MCTS.png')
    
   
    
def process_info():
    with open('mctspns_outcome.pickle','rb') as fp:
        outcome = pkl.load(fp)
    
    for sim_num, record in outcome:  
        total_simulated, total_proven, total_preproven = 0,0,0
        infos = record['info']
        for count in infos:
            total_simulated += count['simulated']
            total_proven += count['proven']
            total_preproven += count['pre-proven']
        
        print(f'Number of Simulations: {sim_num}\n\
                Mean Simulated: {total_simulated/len(infos)}\n\
                Mean Proven : {total_proven/len(infos)}\n\
                Mean Pre-proven: {total_preproven/len(infos)}\n')
        


        

def main():
    # run_trials()
    # plot_figure()
    process_info()

if __name__ == '__main__':
    main()