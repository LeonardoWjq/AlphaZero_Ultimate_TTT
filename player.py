import numpy as np
import torch
import random
import mcts
import mcts_pns
from neural_net import Network
from policy import NNPolicy, RandomPolicy
from termcolor import colored
from neural_mcts import NeuralMCTS
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
    def __init__(self, sim_player,num_simulation = 200, explore = 0.4, store_hist = False) -> None:
        self.sim = sim_player if sim_player else RandomPlayer()
        self.mcts_agent = None
        self.num_sim = num_simulation
        # flag for storing the history of play
        self.store_hist = store_hist
        self.history = []
        self.C = explore

        
    '''
    select a move given a state
    store the history if the flag is True
    '''
    def move(self,state:dict):
        # create the MCTS agent if it does not exist
        if self.mcts_agent is None:
            self.mcts_agent = mcts.MCTS(state,self.sim,exploration_factor=self.C)
            self.mcts_agent.run_simumation(self.num_sim)
            move = self.mcts_agent.get_move()
        else:
            # do a transplantation
            self.mcts_agent.transplant(state)
            self.mcts_agent.run_simumation(self.num_sim)
            move = self.mcts_agent.get_move()
        
        # store history
        if self.store_hist:
            relative_board = state['inner']*state['current']
            self.history.append((relative_board))
        
        return move

    '''
    reset the player for a new game:
    set agent to None
    set step to 0
    clear the history buffer
    '''
    def reset(self):
        self.mcts_agent = None
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
        board = board[None, None]
        board = torch.from_numpy(board)

        # get the probabilities in shape (,81)
        probs = self.network(board)[0]
        
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
        move_index = random.choice(max_move_indices)

        return valid_moves[move_index]


class AlphaZeroPlayer(Player):
    def __init__(self, num_simulation = 200) -> None:
        try:
            model = torch.load('model.pt')
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
        
        
        
class MCTSPNSPlayer(Player):
    # initialize the attributes
    def __init__(self,prior_policy,sim_player,num_simulation = 200) -> None:
        self.pol = prior_policy if prior_policy else RandomPolicy()
        self.sim = sim_player if sim_player else RandomPlayer()
        self.mctspns_agt = None
        self.num_sim = num_simulation
        self.info = {'simulated':0, 'proven':0,'pre-proven':0}
                
    '''
    select a move given a state
    '''
    def move(self,state:dict):
        # create the MCTS agent if it does not exist
        if self.mctspns_agt is None:
            self.mctspns_agt = mcts_pns.MCTSPNS(state, self.pol, self.sim)
            self.mctspns_agt.run_simumation(self.num_sim)
            move,_,key = self.mctspns_agt.get_move()
        else:
            # do a transplantation
            self.mctspns_agt.transplant(state)
            self.mctspns_agt.run_simumation(self.num_sim)
            move,_, key= self.mctspns_agt.get_move()
        
        self.info[key] += 1
        print(move)
        return move
    
    def get_info(self):
        return self.info

    '''
    reset the player for a new game:
    set agent to None
    set step to 0
    clear the history buffer
    '''
    def reset(self):
        self.mctspns_agt = None
        self.info = {'simulated':0, 'proven':0,'pre-proven':0}


class NeuralMCTSPlayer(Player):
    # initialize the attributes
    def __init__(self,sim_player,num_simulation=200,threshold=50,explore = 0.4,is_regression=True) -> None:
        self.sim = sim_player if sim_player else RandomPlayer()
        self.nmcts_agt = None
        self.num_sim = num_simulation
        self.regression = is_regression
        self.threshold = threshold
        self.C = explore
        self.info = {'simulated':0, 'inferred':0}
                
    '''
    select a move given a state
    '''
    def move(self,state:dict):
        # create the MCTS agent if it does not exist
        if self.nmcts_agt is None:
            self.nmcts_agt = NeuralMCTS(state,self.sim,exploration_factor=self.C,threshold=self.threshold,is_regression=self.regression)
            self.nmcts_agt.run_simumation(self.num_sim)
            move,key = self.nmcts_agt.get_move()
        else:
            # do a transplantation
            self.nmcts_agt.transplant(state)
            self.nmcts_agt.run_simumation(self.num_sim)
            move,key= self.nmcts_agt.get_move()
        
        self.info[key] += 1
        return move
    
    def get_info(self):
        return self.info

    '''
    reset the player for a new game:
    set agent to None
    set step to 0
    clear the history buffer
    '''
    def reset(self):
        self.nmcts_agt = None
        self.info = {'simulated':0, 'inferred':0}
        