import pns_tt
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from environment import UltimateTTT
from pns import PNS
from player import RandomPlayer
from TT_util import PROVEN_WIN, AT_LEAST_DRAW, PROVEN_DRAW, AT_MOST_DRAW, PROVEN_LOSS, stats, save, to_list
from termcolor import colored
from tqdm import tqdm


def playout(game:UltimateTTT, num_play = 50, seed = 1):
    '''
    game: the Ultimate TTT game object
    num_play: number of moves to play out
    seed: random seed
    '''
    if seed is not None:
        np.random.seed(seed)
    for _ in range(num_play):
        if game.game_end:
            break
        move = game.make_move()
        game.update_game_state(move)


def generate_entries(start, end, num_move = 65, checkpoint = 1000, verify = False, verbose = 0):
    '''
    generate playing experiences and store proved nodes in the table
    start: start index of iterations
    end: end index of iterations + 1
    num_move: number of moves to play out
    checkpoint: number of moves before saving to the disk
    verify: verify the result from the PNS with TT is consistent without TT
    verbose: 0: doesn't print anything 1: print time and hit rate 2: print everything
    '''
    player = RandomPlayer()
    start_time = time.time()
    assert start < end
    for game_num in tqdm(range(start, end)):
        # set up game
        game = UltimateTTT(player,player,keep_history=False)
        # play the pre-set number of moves
        playout(game, num_move, game_num)
        # alternating target player
        target_player = 1 if game_num %2 == 0 else -1

        '''
        Non exact turn
        '''
        # set up transposition table pns agent
        pnstt_agt = pns_tt.PNSTT(game, target_player, exact=False)
        result_pnstt, _ = pnstt_agt.run()
        if verify:
            # verify with naive pns
            pns_agt = PNS(game, target_player, exact=False)
            result_pns, _ = pns_agt.run()
            assert result_pnstt == result_pns
        
        '''
        Exact turn
        '''
        # set up transposition table pns agent
        pnstt_agt_exact = pns_tt.PNSTT(game, target_player, exact=True)
        result_pnstt, _ = pnstt_agt_exact.run()
        if verify:
            # verify with naive pns
            pns_agt = PNS(game, target_player, exact=True)
            result_pns, _ = pns_agt.run()
            assert result_pnstt == result_pns
            
                
        
        # save the table on checkpoints
        if (game_num + 1) % checkpoint == 0:
            save(pns_tt.TT)
            if verbose == 1 or verbose == 2:
                print(colored(f'Time used so far: {(time.time() - start_time):.2f} s', 'yellow'))
                print(colored('Current hit rate: {:.2%}'.format(pns_tt.hit_num/pns_tt.node_num), 'yellow'))
    
    # save the final outcome
    save(pns_tt.TT) 
    if verbose == 1 or verbose == 2:
        print(colored(f'Total time used: {(time.time() - start_time):.2f} s', 'yellow'))
        print(colored('Overall hit rate: {:.2%}'.format(pns_tt.hit_num/pns_tt.node_num), 'yellow'))
        if verbose == 2:
            print_stats(stats(pns_tt.TT))
        

def prepare_records(num:int, record_type:int):
    '''
    prepare num of records of specified type
    if there are fewer records than num,
    then return all of them
    '''
    # load table
    table = pns_tt.TT
    records = []
    count = 0
    for row in table:
        if row:
            for entry in row:
                if entry[1] == record_type:
                    records.append(entry[0])
                    count += 1

                # break out if enough
                if count == num:
                    return records

    # return all records collected
    return records


def make_exact(num_records = 5000, record_type = AT_LEAST_DRAW, checkpoint=3000):
    '''
    re-play num_records number of games of type record_type to make them exact
    saves the table every checkpoint number of games
    '''
    print(colored('Before:','red'))
    print_stats(stats(pns_tt.TT))
    # extract the wanted records
    records = prepare_records(num_records, record_type)
    player = RandomPlayer()
    for index, state in tqdm(enumerate(records)):
        # replay game from that state
        game  = UltimateTTT(player, player, state, False)
        pnstt_agt = pns_tt.PNSTT(game, state['current'])
        pnstt_agt.run()
        # saving the table
        if (index+1)%checkpoint == 0:
            print(colored('Saving:','yellow'))
            save(pns_tt.TT)
    
    # save the record
    print(colored('Saving:','yellow'))
    save(pns_tt.TT)
    print(colored('After:','red'))
    print_stats(stats(pns_tt.TT))


def print_stats(statistics:dict, plot=False):
    '''
    Print the stats of the transposition table beautifully
    '''
    print(colored(f'Total number of records: {statistics["total"]:,}','green'))
    print(colored(f'Total number of proven wins: {statistics["proven win"]:,}','green'))
    print(colored(f'Total number of at least draws: {statistics["at least draw"]:,}','green'))
    print(colored(f'Total number of proven draws: {statistics["proven draw"]:,}','green'))
    print(colored(f'Total number of at most draws: {statistics["at most draw"]:,}','green'))
    print(colored(f'Total number of proven losses: {statistics["proven loss"]:,}','green'))
    print(colored(f'Mean entry length: {statistics["mean entry length"]:.2f}','green'))
    print(colored(f'Entry length standard deviation: {statistics["entry length std"]:.2f}','green'))
    if plot:
        plt.title('Distribution over Depth')
        plt.plot(statistics['depth distribution'])
        plt.xlabel('Depth')
        plt.ylabel('Count')
        plt.grid()
        plt.show()


def test_running_time(num_game:int = 20, num_playout:int = 60):
    '''
    num_game: number of games to play
    num_playout: number of playout moves before solving
    '''
    player = RandomPlayer()
    total_time = 0
    for i in tqdm(range(num_game)):
        game = UltimateTTT(player,player,keep_history=False)
        playout(game,num_playout,seed=None)
        target_player = 1 if i %2 == 0 else -1

        pnstt_agt = pns_tt.PNSTT(game,target_player,exact=False)
        _, time_used = pnstt_agt.run()
        total_time += time_used

        pnstt_agt = pns_tt.PNSTT(game,target_player,exact=True)
        _, time_used = pnstt_agt.run()
        total_time += time_used
    
    print(colored(f'Total time used: {total_time:.2f} s','yellow'))
    print(colored(f'Average running time: {total_time/num_game:.2f} s', 'yellow'))
    return total_time, total_time/num_game


def make_dataset(is_regression = True):
    table = pns_tt.TT
    # to a list of records first
    record_list = to_list(table)
    inners = []
    outers = []
    outcomes = []
    categories = (PROVEN_WIN, AT_LEAST_DRAW, PROVEN_DRAW, AT_MOST_DRAW, PROVEN_LOSS)

    def legal_move_repr(legal_moves):
        feature_map = np.zeros((9,9))
        for move in legal_moves:
            row = move//9
            col = move%9
            feature_map[row,col] = 1
        return feature_map

    for state, outcome in record_list:
        current_player = state['current']
        inner_board = state['inner']*current_player
        outer_board = state['outer']*current_player
        valid_moves = state['valid_move']

        move_feature = legal_move_repr(valid_moves)
        inner_board = np.concatenate((inner_board[None], move_feature[None]))

        inners.append(inner_board)
        outers.append(outer_board)

        if is_regression:
            # store the scalar record
            outcomes.append(outcome*current_player)
        else:
            # store the category index
            outcomes.append(categories.index(outcome*current_player))


    # to numpy arrays first
    inners = np.array(inners)
    outers = np.array(outers)

    # to tensors
    inners = torch.tensor(inners, dtype=torch.float64)
    outers = torch.tensor(outers, dtype=torch.float64)
    if is_regression:
        outcomes = torch.tensor(outcomes, dtype=torch.float64)
    else:
        outcomes = torch.tensor(outcomes, dtype=torch.int64)
    
    # expand on channel
    # inners = inners[:,None,:]
    # flatten outer board
    outers = outers.view(-1,9)

    if is_regression:
        # expand on label dimension
        outcomes = outcomes[:,None]
        # save dataset
        torch.save((inners,outers,outcomes),'dataset_regression.pt')
    else:
    
        # save dataset
        torch.save((inners,outers,outcomes),'dataset_classification.pt')


    



def main():
    make_dataset(False)
    # print_stats(stats(pns_tt.TT),True)
    # for num in range(63, 62, -1):
    #     print('Rand play:', num)
    #     # take the first 1000 games as tests
    #     generate_entries(start=50000,end=52000,num_move=num,checkpoint=10000,verify=True,verbose=1)
    #     # continue generating
    #     generate_entries(start=52000,end=70000,num_move=num,checkpoint=10000,verify=False,verbose=2)
    # generate_entries(start=120000,end=140000,num_move=63,checkpoint=10000,verify=False,verbose=2)
    
        
   


if __name__ == '__main__':
    main()