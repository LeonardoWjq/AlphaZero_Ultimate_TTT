import numpy as np
from env.macros import *

from mcts.tree_node import TreeNode
from utils.env_utils import equal_state

class MCTS:
    # initialze attributes
    def __init__(self, state:dict, roll_out_player, explore_factor) -> None:
        self.root = TreeNode(state, roll_out_player, explore_factor)
        self.player = roll_out_player
        self.C = explore_factor

    def run_simulation(self, num: int):
        x_win_total = 0
        o_win_total = 0
        tie_total = 0
        for _ in range(num):
            outcome = self.root.simulate()
            if outcome == X_WIN:
                x_win_total += 1
            elif outcome == O_WIN:
                o_win_total += 1
            else:
                tie_total += 1
        
        assert x_win_total + o_win_total + tie_total == num, f'sum of totals is inconsistent with the number of simulations'

        return (x_win_total/num, o_win_total/num, tie_total/num)


    def move_and_truncate(self) -> int:
        '''
        1. select a move from the simulation distribution
        2. transfer the root to the subtree resulted from the move
        return the next move
        '''
        moves, visit_counts = self.root.get_distribution()
        probs = visit_counts/np.sum(visit_counts)
        next_move = np.random.choice(moves, p = probs)

        next_node = self.root.edges[next_move].get_node()
        self.root = next_node

        return next_move
    
    def truncate(self, state: dict) -> None:
        prev_move = state['previous_move']
        try:
            next_node = self.root.edges[prev_move].get_node()
            assert next_node is not None, 'next node is None'
            assert equal_state(next_node.state, state), 'state not equal'
            self.root = next_node
        except (KeyError, AssertionError):
            new_node = TreeNode(state, self.player, self.C)
            self.root = new_node
        
       
    
        
        

        
