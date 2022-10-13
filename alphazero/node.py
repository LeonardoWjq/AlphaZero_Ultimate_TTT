from collections import deque
from copy import copy

import jax.numpy as jnp
import numpy as np
from env.macros import *
from env.ultimate_ttt import UltimateTTT
from jax.random import dirichlet
from utils.alphazero_utils import (compute_puct_score, create_feature,
                                   get_val_and_pol)
from utils.env_utils import get_valid_moves, inner_to_outer

from alphazero.edge import Edge


class Node:
    '''
    game_state is already in history
    '''

    def __init__(self, game_state: dict, history: deque, forward_func, explore_factor: float, is_root: bool, rand_key, alpha: float, epsilon: float) -> None:
        self.state: dict = game_state
        self.C: float = explore_factor
        self.forward = forward_func
        self.current_player: int = game_state['current_player']
        self.history: deque = copy(history)
        self.outcome: int = game_state['outcome']

        if self.outcome == INCOMPLETE:
            inner_board = game_state['inner_board']
            outer_board = inner_to_outer(inner_board)
            prev_move = game_state['previous_move']
            valid_moves = get_valid_moves(inner_board, outer_board, prev_move)

            self.feature = create_feature(history, self.current_player)

            state_value, priors = get_val_and_pol(
                forward_func, self.feature, valid_moves)

            self.state_val: float = state_value

            self.edges: dict = {}
            if is_root:
                alpha_vec = jnp.ones(len(valid_moves))*alpha
                dirichlet_noises = dirichlet(rand_key, alpha_vec)
                for move, prior, noise in zip(valid_moves, priors, dirichlet_noises):
                    self.edges[move] = Edge(prior_prob=(
                        1-epsilon)*prior + epsilon*noise)
            else:
                for move, prior in zip(valid_moves, priors):
                    self.edges[move] = Edge(prior_prob=prior)

    def unroll(self) -> int:
        '''
        unroll one step
        '''

        if self.outcome == INCOMPLETE:
            return self.state_val
        else:
            if self.outcome == TIE:
                return 0.0
            else:
                if self.current_player == X:
                    return 1.0 if self.outcome == X_WIN else -1.0
                else:
                    return 1.0 if self.outcome == O_WIN else -1.0

    def get_max_move(self) -> int:
        '''
        get the best move according to the PUCT score
        '''
        total_visit = 0
        for edge in self.edges.values():
            total_visit += edge.get_visit_count()

        max_value = float('-inf')
        max_moves = []
        for move, edge in self.edges.items():
            score = compute_puct_score(edge, total_visit, self.C)
            if score > max_value:
                max_value = score
                max_moves = [move]
            elif score == max_value:
                max_moves.append(move)

        return np.random.choice(max_moves)

    def simulate(self):
        '''
        perform tree policy and back up
        '''
        if self.outcome != INCOMPLETE:
            if self.outcome == TIE:
                return 0.0
            else:
                if self.current_player == X:
                    return 1.0 if self.outcome == X_WIN else -1.0
                else:
                    return 1.0 if self.outcome == O_WIN else -1.0

        move = self.get_max_move()
        edge = self.edges[move]
        if edge.get_node() is None:  # grow the edge
            game = UltimateTTT(None, None, self.state)
            game.update_state(move)
            next_state = game.get_state()
            next_hist = copy(self.history)
            next_hist.appendleft(next_state)

            # create new tree node
            new_node = Node(next_state, next_hist, self.forward,
                            self.C, False, None, None, None)
            edge.set_node(new_node)

            score = new_node.unroll()
        else:  # simulate next node
            score = edge.get_node().simulate()

        # increment count
        edge.increment_visit_count()
        # score is respect to opponent's turn, thus shoud negate it
        edge.add_action_val(-score)
        return -score

    def get_feature(self):
        return self.feature

    def get_dist(self):
        moves = []
        visit_count = []

        for move, edge in self.edges.items():
            moves.append(move)
            visit_count.append(edge.get_visit_count())

        visit_count = np.array(visit_count)

        return moves, visit_count
