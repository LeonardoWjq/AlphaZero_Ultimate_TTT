import numpy as np
import torch
import random
import mcts
import mcts_pns
from neural_net import Network
from policy import NNPolicy, RandomPolicy
from termcolor import colored
from neural_mcts import NeuralMCTS






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
        