import numpy as np
class Player:
    def move(self, board:np.array, valid_moves:list)->int:
        pass

class RandomPlayer(Player):
    def move(self, board, valid_moves):
        return np.random.choice(valid_moves)

