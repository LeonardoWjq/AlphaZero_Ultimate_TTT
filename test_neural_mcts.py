from player import NeuralMCTSPlayer, MCTSPlayer
from environment import UltimateTTT as UTTT
from tqdm import tqdm
import pickle as pkl
def main():
    NMCTS_win = 0
    MCTS_win = 0
    Draw = 0
    p1 = NeuralMCTSPlayer(None,threshold=55)
    p2 = MCTSPlayer(None,None)
    for num in tqdm(range(100)):
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

if __name__ == '__main__':
    main()
