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
    # initialize the attributes
    def __init__(self, num_simulation = 300, store_hist = False) -> None:
        self.mcts_agent = None
        self.num_sim = num_simulation
        # store the number of step in one game
        self.step = 0
        # flag for storing the history of play
        self.store_hist = store_hist
        self.history = []

    '''
    select a temperature parameter based on the number of steps played so far in the game
    '''
    def select_temp(self):
        if self.step < 10:
            return 2
        elif self.step < 20:
            return 1
        else:
            return 0.5
        
    '''
    select a move given a state
    store the history if the flag is True
    '''
    def move(self,state:dict):
        temperature = self.select_temp()
        # create the MCTS agent if it does not exist
        if self.mcts_agent is None:
            pol = RandomPolicy()
            sim_player = RandomPlayer()
            self.mcts_agent = mcts.MCTS(state,pol,sim_player)
            self.mcts_agent.run_simumation(self.num_sim)
            move,probs = self.mcts_agent.get_move(temp=temperature)
        else:
            # do a transplantation
            self.mcts_agent.transplant(state)
            self.mcts_agent.run_simumation(self.num_sim)
            move,probs = self.mcts_agent.get_move(temp=temperature)
        
        # store history
        if self.store_hist:
            self.history.append((state, probs))
        
        return move

    '''
    reset the player for a new game:
    set agent to None
    set step to 0
    clear the history buffer
    '''
    def reset(self):
        self.mcts_agent = None
        self.step = 0
        self.history = []
    
    '''
    return the history buffer
    '''
    def get_history(self):
        return self.history
