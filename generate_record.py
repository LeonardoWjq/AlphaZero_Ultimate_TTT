import pns_tt
import numpy as np
import time
from environment import UltimateTTT
from pns import PNS
from player import RandomPlayer
from TT_util import stats, save
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
        pnstt_agt = pns_tt.PNSTT(game, target_player, exact=True)
        result_pnstt, _ = pnstt_agt.run()
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
        


def print_stats(statistics:dict):
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


def main():
    generate_entries(0,50000,num_move=61,checkpoint=5000,verbose=2)
   


if __name__ == '__main__':
    main()