from collections import namedtuple
from environment import UltimateTTT
from player import MCTSPlayer

'''
given a history list and the winner of the game
output a list of named tuple that is useful for training the neural network
'''
def to_dataset(history, winner):
    Data = namedtuple('data',['board', 'probs', 'score'])
    dataset = []
    for state, probs in history:
        if not state['game_end']:
            relative_board = state['inner']*state['current']
            score = 0 if winner == 2 else winner*state['current']
            data = Data(relative_board, probs, score)
            dataset.append(data)
    
    return dataset


'''
let the player play game with itself
input a player and its copy
specify the number of games
'''
def self_play(player1:MCTSPlayer,player2:MCTSPlayer,num_games=10):
    for index in range(num_games):
        game = UltimateTTT(player1,player2)
        game.play()
        hist1 = player1.get_history()
        hist2 = player2.get_history()
        winner = game.winner
        
        data1 = to_dataset(hist1,winner)
        data2 = to_dataset(hist2,winner)
        dataset = data1 + data2


def main():
    player1 = MCTSPlayer(num_simulation=100,store_hist=True)
    player2 = MCTSPlayer(num_simulation=100,store_hist=True)
    self_play(player1, player2,1)

if __name__ == '__main__':
    main()