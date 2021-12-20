import numpy as np
import torch
from network import Network

class Policy:
    def get_probs(self, state:dict):
        pass

class NNPolicy(Policy):
    def __init__(self, NNet:Network):
        self.network = NNet
    
    '''
    take state as an input
    output the re-normalized probabilities of each valid move in a list
    '''
    def get_probs(self, state: dict):
        valid_moves = state['valid_move']
        board_state = torch.Tensor(state['inner']*state['current'])
        move_probs = self.network(board_state[None, None])

        valid_move_probs = []
        total_valid_probs = 0

        for move in valid_moves:
            valid_prob = move_probs[0][0][move].item()
            valid_move_probs.append(valid_prob)
            total_valid_probs += valid_prob
        
        for i, _ in enumerate(valid_move_probs):
            valid_move_probs[i] /= total_valid_probs

        return valid_move_probs


class RandomPolicy(Policy):
    '''
    input a state
    output a vector of equal probabilities corresponding to the dimension of valid moves
    '''
    def get_probs(self, state: dict):
        length = len(state['valid_move'])
        probs = np.ones(length)/length
        return probs
