import numpy as np
import mcts
from policy import RandomPolicy
class Player:
    def move(self, state:dict)->int:
        pass

class RandomPlayer(Player):
    # return a random valid move
    def move(self,state:dict)->int:
        valid_moves = state['valid_move']
        return np.random.choice(valid_moves)

class HumanPlayer(Player):
    def move(self, state:dict)->int:
        # get row number as an integer
        row_valid = False
        while not row_valid:
            row = input("Please enter a row number: ")
            try:
                row_num = int(row)
                row_valid = True
            except ValueError:
                print("Invalid input. Please enter an integer as specified in the valid moves.\n")
        
        # get column number as an integer
        col_valid = False
        while not col_valid:
            col = input("Please enter a column number: ")
            try:
                col_num = int(col)
                col_valid = True
            except ValueError:
                print("Invalid input. Please enter an integer as specified in the valid moves.\n")
        
        # calculate row order position
        position = int(row_num)*9 + int(col_num)
        
        return position


class MCTSPlayer(Player):
    def __init__(self, num_simulation = 300) -> None:
        self.num_sim = num_simulation
    def move(self,state:dict):
        pol = RandomPolicy()
        sim_player = RandomPlayer()
        mcts_agent = mcts.MCTS(state,pol,sim_player)
        mcts_agent.run_simumation(self.num_sim)
        return mcts_agent.get_move()
