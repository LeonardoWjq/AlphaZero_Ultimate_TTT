
import numpy as np
from env.macros import *
from env.ultimate_ttt import UltimateTTT
from utils.env_utils import get_valid_moves, inner_to_outer

from mcts.edge import Edge


class TreeNode:
    '''
    represent a node in the search tree
    maintains a single state of the game
    contains the functionalities for performing simulations
    '''

    def __init__(self, state: dict, roll_out_player, explore_factor) -> None:
        self.state = state
        self.player = roll_out_player
        self.C = explore_factor

        inner_board = state['inner_board']
        outer_board = inner_to_outer(inner_board)
        prev_move = state['previous_move']
        valid_moves = get_valid_moves(inner_board, outer_board, prev_move)

        self.edges = {}
        for move in valid_moves:
            self.edges[move] = Edge()

        self.is_terminal = not (state['outcome'] == INCOMPLETE)

    def unroll(self) -> int:
        '''
        let the simulation players play until the end
        return the outcome of the game
        '''
        if self.is_terminal:
            return self.state['outcome']

        game = UltimateTTT(self.player, self.player, self.state)
        game.play()
        return game.outcome

    def get_max_move(self) -> int:
        '''
        get the best move according to the following rules:
        1. If some moves have never been selected before, then
           they are prioritized.
        2. If all the moves have been selected at least once,
           then the ones with the (same) highest UCT score are candidates.
        2. When ties happen, randomly choose a move.
        '''
        total_visit = 0
        for edge in self.edges.values():
            total_visit += edge.get_visit_count()

        max_value = float('-inf')
        max_moves = []
        for move, edge in self.edges.items():
            if edge.get_visit_count() == 0:  # the edge has never been visited before
                # the previous ones are all visited edges
                if max_value < float('inf'):
                    max_value = float('inf')
                    max_moves = [move]
                else:
                    max_moves.append(move)
            else:  # the edge has been visited before
                q = edge.get_avg()
                ucb = np.sqrt(np.log(total_visit)/edge.get_visit_count())
                value = q + self.C*ucb
                if value > max_value:
                    max_value = value
                    max_moves = [move]
                elif value == max_value:
                    max_moves.append(move)

        return np.random.choice(max_moves)

    def simulate(self):
        '''
        perform tree policy and back up
        returns the outcome of simulated game
        '''
        if self.is_terminal:
            return self.state['outcome']

        move = self.get_max_move()
        edge = self.edges[move]
        if edge.get_node() is None:  # grow the edge
            game = UltimateTTT(self.player, self.player, self.state)
            game.update_state(move)
            next_state = game.get_state()

            # create new tree node
            new_node = TreeNode(next_state, self.player, self.C)
            edge.set_node(new_node)

            outcome = new_node.unroll()
        else:  # simulate next node
            outcome = edge.get_node().simulate()

        # update statistics
        edge.increment_visit_count()
        if outcome != TIE:
            current_player = self.state['current_player']
            if outcome == X_WIN:
                edge.add_val(1.) if current_player == X else edge.add_val(-1.)
            else:
                edge.add_val(1.) if current_player == O else edge.add_val(-1.)

        return outcome

    def get_distribution(self):
        '''
        return (array of moves, array of visit counts)
        '''
        moves = []
        visits = []
        for move, edge in self.edges.items():
            moves.append(move)
            visits.append(edge.get_visit_count())

        return np.array(moves), np.array(visits)
