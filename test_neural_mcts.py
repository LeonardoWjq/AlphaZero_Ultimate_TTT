from player import NeuralMCTSPlayer, MCTSPlayer
from environment import UltimateTTT as UTTT
from tqdm import tqdm
import pickle as pkl

def test_performance(thresholds, is_regression=True, num_game = 100):
    assert num_game > 0

    for thd in thresholds:
        assert thd >= 0

        nmcts_win = 0
        mcts_win = 0
        draw = 0
        sum_prop = 0
        
        p1 = NeuralMCTSPlayer(None,threshold=thd,explore=0.3,is_regression=is_regression)
        p2 = MCTSPlayer(None,explore=0.3)

        for num in tqdm(range(num_game)):
            if num%2 == 0:
                game = UTTT(p1,p2,keep_history=False)
                game.play()

                if game.winner == 1:
                    nmcts_win += 1
                elif game.winner == 2:
                    draw += 1
                elif game.winner == -1:
                    mcts_win += 1
                
            else:
                game = UTTT(p2,p1,keep_history=False)
                game.play()

                if game.winner == 1:
                    mcts_win += 1
                elif game.winner == 2:
                    draw += 1
                elif game.winner == -1:
                    nmcts_win += 1
                    
            info = p1.get_info()
            sum_prop +=  info['inferred']/(info['simulated'] + info['inferred'])

            p1.reset()
            p2.reset()

        if is_regression:
            with open(f'test_results/test_nmcts_regression_{thd}.pickle','wb') as fp:
                pkl.dump((nmcts_win,mcts_win,draw,sum_prop/num_game),fp)
        else:
            with open(f'test_results/test_nmcts_classification_{thd}.pickle','wb') as fp:
                pkl.dump((nmcts_win,mcts_win,draw,sum_prop/num_game),fp)

def print_result(thresholds, is_regression=True):
    if is_regression:
        for thd in thresholds:
            with open(f'test_results/test_nmcts_regression_{thd}.pickle','rb') as fp:
                win, loss, draw, proportion = pkl.load(fp)
                print(f'Threshold:{thd} Neural(regression) MCTS\tWin:{win}\tLoss:{loss}\tDraw:{draw}\tInference Proportion:{proportion}')
    else:
        for thd in thresholds:
            with open(f'test_results/test_nmcts_classification_{thd}.pickle','rb') as fp:
                win, loss, draw, proportion = pkl.load(fp)
                print(f'Threshold:{thd} Neural(classification) MCTS\tWin:{win}\tLoss:{loss}\tDraw:{draw}\tInference Proportion:{proportion}')




def main():
    test_performance(thresholds=(45,55,65),is_regression=False,num_game=100)
    print_result((45,55,65),is_regression=False)
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
