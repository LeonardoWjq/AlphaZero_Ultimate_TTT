from player import NeuralMCTSPlayer, MCTSPlayer
from environment import UltimateTTT as UTTT
from tqdm import tqdm
import pickle as pkl

def test_performance():
    NMCTS_win = 0
    MCTS_win = 0
    Draw = 0
    p1 = NeuralMCTSPlayer(None,threshold=40,explore=0.3,is_regression=False)
    p2 = MCTSPlayer(None,explore=0.3)
    for num in tqdm(range(200)):
        if num%2 == 0:
            game = UTTT(p1,p2,keep_history=False)
            game.play()
            print(p1.get_info())
            if game.winner == 1:
                NMCTS_win += 1
            elif game.winner == 2:
                Draw += 1
            elif game.winner == -1:
                MCTS_win += 1
            
        else:
            game = UTTT(p2,p1,keep_history=False)
            game.play()
            print(p1.get_info())
            if game.winner == 1:
                MCTS_win += 1
            elif game.winner == 2:
                Draw += 1
            elif game.winner == -1:
                NMCTS_win += 1

        p1.reset()
        p2.reset()
    
    print(f'NMCTS Win:{NMCTS_win}, MCTS Win:{MCTS_win}, Draw:{Draw}')
    with open('test_nmcts_result.pickle','wb') as fp:
        pkl.dump((NMCTS_win,MCTS_win,Draw),fp)

def main():
    test_performance()
    # p1_win = 0
    # p2_win = 0
    # draw = 0
    # p1 = MCTSPlayer(None,explore=0.3)
    # p2 = MCTSPlayer(None)
    # for i in tqdm(range(100)):
    #     if i%2 == 0:
    #         game = UTTT(p1,p2,keep_history=False)
    #         game.play()
    #         if game.winner == 1:
    #             p1_win += 1
    #         elif game.winner == -1:
    #             p2_win += 1
    #         else:
    #             draw += 1
    #     else:
    #         game = UTTT(p2,p1,keep_history=False)
    #         game.play()
    #         if game.winner == 1:
    #             p2_win += 1
    #         elif game.winner == -1:
    #             p1_win += 1
    #         else:
    #             draw += 1
    #     p1.reset()
    #     p2.reset()
    # print('P1 Win:',p1_win,'P2 Win:',p2_win,'Draw:',draw)


if __name__ == '__main__':
    main()
