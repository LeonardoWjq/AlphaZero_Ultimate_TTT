from unicodedata import name
from env.ultimate_ttt import UltimateTTT
from env.macros import *
from typing import TypeVar
from collections import namedtuple

Node = TypeVar('Node')
Edge = namedtuple('Edge', ['move', 'child'])

class Node:
    def __init__(self, state: dict, parent: Node, root_player: int, bounded: bool) -> None:
        self.state = state
        self.parent = parent
        self.root_player = root_player
        self.is_leaf_node = True
        self.is_or_node = state['current_player'] == root_player
        self.bounded = bounded

        # initialize the proof number and disproof number
        if state['outcome'] == X_WIN:
            if self.root_player == X:
                self.pn = 0
                self.dn = float('inf')
            else:
                self.pn = float('inf')
                self.dn = 0
        elif state['outcome'] == O_WIN:
            if self.root_player == O:
                self.pn = 0
                self.dn = float('inf')
            else:
                self.pn = float('inf')
                self.dn = 0
        elif state['outcome'] == TIE:
            if bounded:
                self.pn = 0
                self.dn = float('inf')
            else:
                self.pn = float('inf')
                self.dn = 0
        else:
            self.pn = 1
            self.dn = 1

        self.children = []

    def expand(self):
        '''
        expand the node by initializing its children
        '''
        assert self.state['outcome'] == INCOMPLETE, 'cannot expand terminal state'
        assert len(self.children) == 0, 'node already expanded'

        game = UltimateTTT(None, None, self.state)
        valid_moves = game.next_valid_moves
        for move in valid_moves:
            game.update_state(move)
            child = Node(game.get_state(), self, self.root_player, self.bounded)
            self.children.append(Edge(move, child))
            game.undo()

        self.is_leaf_node = False

    def get_numbers(self):
        return self.pn, self.dn

    def find_equal_child_pn(self, nodes, parent_pn):
        for move, child in nodes:
            pn, _ = child.get_numbers()
            if pn == parent_pn:
                return move, child

        raise RuntimeError(
            'No child with the same proof number has been found.')

    def find_equal_child_dn(self, nodes, parent_dn):
        for move, child in nodes:
            _, dn = child.get_numbers()
            if dn == parent_dn:
                return move, child

        raise RuntimeError(
            'No child with the same disproof number has been found.')

    def select_MPN(self):
        '''
        return the minimum proof or disproof node
        '''
        node = self
        while not node.is_leaf_node:
            pn, dn = node.get_numbers()
            if node.is_or_node:
                _, node = self.find_equal_child_pn(node.children, pn)
            else:
                _, node = self.find_equal_child_dn(node.children, dn)
        return node

    def update_proof_number(self):
        '''
        update the proof numbers and disproof numbers from the current node and up
        '''
        child_pns = []
        child_dns = []
        for _, child in self.children:
            pn, dn = child.get_numbers()
            child_pns.append(pn)
            child_dns.append(dn)

        # is or node
        if self.is_or_node:
            self.pn = min(child_pns)
            self.dn = sum(child_dns)

        # is and node
        else:
            self.pn = sum(child_pns)
            self.dn = min(child_dns)

        # recursively update parent proof/disproof numbers
        if self.parent is not None:
            self.parent.update_proof_number()


class PNS:
    def __init__(self, state: dict, bounded: bool) -> None:
        self.game = UltimateTTT(None, None, state)
        root_player = state['current_player']
        self.root = Node(state, None, root_player, bounded)

    def _search(self):
        '''
        perform the proof number search
        return the evaluation for the root player
        '''
        pn, dn = self.root.get_numbers()
        while pn != 0 and dn != 0:
            mpn = self.root.select_MPN()
            mpn.expand()
            mpn.update_proof_number()
            pn, dn = self.root.get_numbers()
        return pn == 0

    def run(self):
        result: bool = self._search()
        move: int = self._next_best_move()
        return result, move

    def _next_best_move(self):
        '''
        return the best move to make for the root player
        '''
        assert not self.root.is_leaf_node, 'cannot move on terminal states'

        pn, _ = self.root.get_numbers()
        move, _ = self.root.find_equal_child_pn(
            self.root.children, pn)  # root player is always an or node

        return move

    # def print_trace(self):
    #     '''
    #     print out the trace that leads to the outcome of search
    #     '''
    #     node = self.root
    #     self.game.display_board()
    #     self.game.display_board('outer')
    #     while not node.is_leaf_node:
    #         pn, dn = node.get_numbers()
    #         player = 'x' if node.state['current'] == 1 else 'o'
    #         if node.is_or_node:
    #             move, node = node.find_equal_child_pn(node.children,pn)
    #         else:
    #             move, node = node.find_equal_child_dn(node.children,dn)

    #         row, col = move//9, move%9
    #         print(f'Player {player} made the move {row, col}')
    #         game = UltimateTTT(None,None,node.state)
    #         game.display_board()
    #         game.display_board('outer')
