import numpy as np
import torch
import random
import mcts
from network import Network
from policy import NNPolicy, RandomPolicy
from termcolor import colored
class Player:
    def move(self, state: dict)->int:
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
    def __init__(self,prior_policy,sim_player,num_simulation = 200, store_hist = False) -> None:
        self.pol = prior_policy
        self.sim = sim_player
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
            self.mcts_agent = mcts.MCTS(state,self.pol,self.sim)
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


class NNPlayer(Player):
    def __init__(self, NNet:Network) -> None:
        self.network = NNet
    
    
    def move(self, state: dict) -> int:
        # board relative to the current player 1 for itself, -1 for its opponent
        board = state['inner']*state['current']
        board = board[None, None, :]
        board = torch.from_numpy(board).float()

        # get the probabilities in shape (,81)
        probs = self.network(board)[0][0]
        valid_moves = state['valid_move']
        valid_probs = probs[valid_moves]
        # normalize valid probabilities so that they sum to 1
        sum = torch.sum(valid_probs)
        if (sum != 0):
            valid_probs = valid_probs/sum

    
        max_move_indices = []
        max_move_prob = 0

        for i, prob in enumerate(valid_probs):
            if prob > max_move_prob:
                max_move_prob = prob
                max_move_indices = [i]
            elif prob == max_move_prob:
                max_move_indices.append(i)
        
        # randomly choose one to break ties
        # if (len(max_move_indices) == 0):
        #     print("NULL")
        #     print(valid_moves)
        #     print(valid_probs1)
        #     print(valid_probs)
        #     print(probs)
        move_index = random.choice(max_move_indices)

        return valid_moves[move_index]


class AlphaZeroPlayer(Player):
    def __init__(self, model_num, num_simulation = 200) -> None:
        try:
            model = torch.load(f'./models/model_{model_num}.pt')
            print(colored('Neural network model loaded successfully.','green'))
        except FileNotFoundError:
            model = Network()
            print(colored('Warning: loading neural network model failed. Initializing a random network instead.','yellow'))
        
        pol = NNPolicy(model)
        sim = NNPlayer(model)
        self.player = MCTSPlayer(pol,sim,num_simulation)

    def move(self, state:dict):
        return self.player.move(state)
    
    def reset(self):
        self.player.reset()
        
        
        
        