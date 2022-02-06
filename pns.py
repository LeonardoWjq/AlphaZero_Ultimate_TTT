from environment import UltimateTTT
from player import RandomPlayer
from termcolor import colored
import time
class Node:
    def __init__(self, state, parent, root_player) -> None:

        self.state = state
        self.parent = parent
        self.root_player = root_player
        self.is_leaf_node = True
        self.is_or_node = state['current'] == root_player
        
        if state['game_end']:
            if self.root_player == state['winner']:
                self.pn = 0
                self.dn = float('inf')
            else:
                self.pn = float('inf')
                self.dn = 0
        else:
            self.pn = 1
            self.dn = 1

        self.children = []
        
    '''
    Expand the node by initializing its children
    '''
    def expand(self):
        if self.state['game_end']:
            print(colored("Cannot expand leaf nodes that are terminal states!", "red"))
            return
        if len(self.children) != 0:
            print(colored('Error: Node already expanded!','red'))
            return

        game = UltimateTTT(None,None,self.state)
        legal_moves = self.state['valid_move']
        for move in legal_moves:
            game.update_game_state(move)
            self.children.append((move, Node(game.get_state(),self,self.root_player)))
            game.undo()
        
        self.is_leaf_node = False
    
    '''
    return (proof number, disproof number)
    '''
    def get_numbers(self):
        return self.pn, self.dn
    
    
    def find_equal_child_pn(self, nodes, parent_pn):
        for move, child in nodes:
            pn,_ = child.get_numbers()
            if pn == parent_pn:
                return move, child
        
        print(colored('Error: Node not found!', 'red'))
    
    def find_equal_child_dn(self, nodes, parent_dn):
        for move, child in nodes:
            _,dn = child.get_numbers()
            if dn == parent_dn:
                return move, child
        
        print(colored('Error: Node not found!', 'red'))

    '''
    return the minimum proof node
    '''
    def select_MPN(self):
        node = self
        while not node.is_leaf_node:
            pn,dn = node.get_numbers()
            if node.is_or_node:
                _,node = self.find_equal_child_pn(node.children, pn)
            else:
                _,node = self.find_equal_child_dn(node.children, dn)
        return node
    
    '''
    update the proof number from the current node and up
    '''
    def update_proof_number(self):
        child_pns = []
        child_dns = []
        for _,child in self.children:
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
    def __init__(self, game:UltimateTTT, player:int, num_rand = 40) -> None:
        self.game = game
        for _ in range(num_rand):
            if self.game.game_end:
                break
            move = self.game.make_move()
            self.game.update_game_state(move)

        self.root = Node(self.game.get_state(), None, player)
    
    '''
    perform the proof number search
    return whether the player to be checked is winning
    '''
    def search(self):
        start_time = time.time()
        pn, dn = self.root.get_numbers()
        while pn != 0 and dn != 0 :
            mpn = self.root.select_MPN()
            mpn.expand()
            mpn.update_proof_number()
            pn, dn = self.root.get_numbers()
        print(f'Time to run Proof Number Search: {time.time() - start_time}')
        return pn == 0

    '''
    print out the trace that leads to the outcome of search
    '''
    def print_trace(self):
        node = self.root
        self.game.display_board()
        self.game.display_board('outer')
        while not node.is_leaf_node:
            pn, dn = node.get_numbers()
            player = 'x' if node.state['current'] == 1 else 'o'
            if node.is_or_node:
                move, node = node.find_equal_child_pn(node.children,pn)
            else:
                move, node = node.find_equal_child_dn(node.children,dn)
            
            row, col = move//9, move%9
            print(f'Player {player} made the move {row, col}')
            game = UltimateTTT(None,None,node.state)
            game.display_board()
            game.display_board('outer')


def main():
    p1 = RandomPlayer()
    p2 = RandomPlayer()
    test_game = UltimateTTT(p1,p2)
    tree = PNS(test_game,1,50)
    res = tree.search()
    print('Winning.') if res else print('Not Winning.')
    tree.print_trace()

if __name__ == '__main__':
    main()