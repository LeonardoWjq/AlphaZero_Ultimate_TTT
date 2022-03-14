import pns_tt
import numpy as np
import pickle as pkl
import time
from environment import UltimateTTT
from pns import PNS
from player import RandomPlayer
from TT_util import stats

def random_play(game:UltimateTTT, num_play = 50, seed = 1):
    np.random.seed(seed)
    for _ in range(num_play):
        if game.game_end:
            break
        move = game.make_move()
        game.update_game_state(move)

def generate_entries(num_game = 50, rand_play = 50, start_seed = 0):
    player = RandomPlayer()
    for i in range(num_game):
        game = UltimateTTT(player,player,keep_history=False)
        random_play(game, rand_play, start_seed + i)
        target = 1 if i%2 == 0 else -1
        pnstt_agent = pns_tt.PNSTT(game, target, exact=True)


def main():
    player = RandomPlayer()
    start = time.time()
    for i in range(20,40):
        game = UltimateTTT(player, player, keep_history=False)
        random_play(game, seed = i)
        target = 1 if i%2 == 0 else -1
        tt_agent = pns_tt.PNSTT(game, target, exact=True)
        # pns_agent = PNS(game,target, exact= True)
        result_tt,_ = tt_agent.run()
        # result_pn,_ = pns_agent.run()
        # assert result_tt == result_pn
        print(pns_tt.hit_num, pns_tt.node_num)
        print('hit rate','{:.2%}'.format(pns_tt.hit_num/pns_tt.node_num))
        tt_agent = pns_tt.PNSTT(game, target, exact=False)
        # pns_agent = PNS(game,target, exact= False)
        result_tt,_ = tt_agent.run()
        # result_pn,_ = pns_agent.run()
        # assert result_tt == result_pn
        print(pns_tt.hit_num, pns_tt.node_num)
        print('hit rate','{:.2%}'.format(pns_tt.hit_num/pns_tt.node_num))
    print('Time used:', time.time() - start)
    with open('tt.pickle','wb') as fp:
        pkl.dump(pns_tt.TT, fp)


if __name__ == '__main__':
    main()