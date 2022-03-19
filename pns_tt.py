from TT_util import (PROVEN_WIN, PROVEN_DRAW, PROVEN_LOSS, AT_LEAST_DRAW, AT_MOST_DRAW,
                    hash_func, lookup, store, load, save)
from pns import Node, PNS
from termcolor import colored
from environment import UltimateTTT 

# transposition table
TT = load()
node_num = 0
hit_num = 0
class NodeTT(Node):
    def __init__(self, state, parent, root_player, exact) -> None:
        global TT, node_num, hit_num
        assert root_player == 1 or root_player == -1
        super().__init__(state, parent, root_player, exact)

        # increment node number by one every time it's created
        node_num += 1

        self.key = hash_func(state)

        same_player = state['current']*root_player

        outcome = lookup(TT, self.key, state)
        # proven node for the root player
        if self.pn == 0:
            # proven win for current player if it is the same as the root player
            # otherwise proven loss
            if exact:
                store(self.key, TT, state, PROVEN_WIN*same_player)
            # at least draw for current player if it is the same as the root
            # otherwise at most draw
            else:
                store(self.key, TT, state, AT_LEAST_DRAW*same_player)
            
            # no need to find it in table
            outcome = None
        elif self.dn == 0:
            # Cannot win. Could draw or lose if the current player is
            # the same as the root. Otherwise, the current player is
            # at least drawing 
            if exact:
                store(self.key, TT, state, AT_MOST_DRAW*same_player)
            # Cannot win or tie. Bound to be proven loss if the players
            # are the same. Otherwise, the current player is winning
            else:
                store(self.key, TT, state, PROVEN_LOSS*same_player)
            
            # no need to find it in table
            outcome = None


        # found outcome in table
        if outcome is not None:
            # increament hit number
            hit_num += 1
            # if the root player is the same as the current player
            if root_player == state['current']:
                # seeking exact result
                if exact:
                    if outcome == PROVEN_WIN:
                        self.pn = 0
                        self.dn = float('inf')
                    # all other conditions don't satisfy an exact win
                    else:
                        self.pn = float('inf')
                        self.dn = 0
                # seeking bounded result
                else:
                    if outcome == PROVEN_WIN or outcome == AT_LEAST_DRAW or outcome == PROVEN_DRAW:
                        self.pn = 0
                        self.dn = float('inf')
                    elif outcome == PROVEN_LOSS:
                        self.pn = float('inf')
                        self.dn = 0
                    '''
                    Cannot say about at most draw, it could either prove or disprove the bounded case
                    '''
            # the root is not the same as the current player
            else:
                # seek exact win
                if exact:
                    # proven loss for the current player is a proven win for the
                    # root player 
                    if outcome == PROVEN_LOSS:
                        self.pn = 0
                        self.dn = float('inf')
                    else:
                        self.pn = float('inf')
                        self.dn = 0
                # seeking bounded result
                else:
                    if outcome == PROVEN_LOSS or outcome == PROVEN_DRAW or outcome == AT_MOST_DRAW:
                        self.pn = 0
                        self.dn = float('inf')
                    elif outcome == PROVEN_WIN:
                        self.pn = float('inf')
                        self.dn = 0
                    '''
                    Cannot say about at least draw, it could either prove or disprove the bounded case
                    '''
    
    '''
    Override: Expand the node by initializing its children
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
            self.children.append((move, NodeTT(game.get_state(),self,self.root_player, self.exact)))
            game.undo()
        
        self.is_leaf_node = False

    '''
    Override: Update the proof number from the current node and up
    Update the table if the node is proven or disproven
    '''
    def update_proof_number(self):
        global TT
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
        
        same_player = self.state['current']*self.root_player
        if self.pn == 0:
            # proven win for current player if it is the same as the root player
            # otherwise proven loss
            if self.exact:
                store(self.key, TT, self.state, PROVEN_WIN*same_player)
            # at least draw for current player if it is the same as the root
            # otherwise at most draw
            else:
                store(self.key, TT, self.state, AT_LEAST_DRAW*same_player)
        elif self.dn == 0:
            # Cannot win. Could draw or lose if the current player is
            # the same as the root. Otherwise, the current player is
            # at least drawing 
            if self.exact:
                store(self.key, TT, self.state, AT_MOST_DRAW*same_player)
            # Cannot win or tie. Bound to be proven loss if the players
            # are the same. Otherwise, the current player is winning
            else:
                store(self.key, TT, self.state, PROVEN_LOSS*same_player)
            
        # recursively update parent proof/disproof numbers
        if self.parent is not None:
            self.parent.update_proof_number()

class PNSTT(PNS):
    def __init__(self, game: UltimateTTT, target_player=1, exact=True) -> None:
        self.game = game
        self.root_player = target_player
        self.root = NodeTT(game.get_state(), None, target_player, exact)
    
    '''
    generate the playing trajectory (state, key) that leads to a proof or disproof
    '''
    def generate_trajectory(self):
        trajectory = []
        node = self.root
        trajectory.append((node.state, node.key))
        while not node.is_leaf_node:
            pn, dn = node.get_numbers()
            if node.is_or_node:
                _, node = node.find_equal_child_pn(node.children,pn)
            else:
                _, node = node.find_equal_child_dn(node.children,dn)
            trajectory.append((node.state, node.key))
        return trajectory
    
    # Override
    def next_best_move(self):
        '''
        return the best move if there is one
        '''
        root_node = self.root
        # If it's leaf node then the best move is unknown
        if root_node.is_leaf_node:
            max_score = -10
            max_move = None
            incomplete = False
            for move in self.game.next_valid_move:
                game = UltimateTTT(None, None, self.game.get_state(),False)
                game.update_board(move)
                state = game.get_state()

                # result of the opponent move
                res = lookup(TT, hash_func(state), state)
                if res is None:
                    # Did not find the record from the table
                    incomplete = True
                elif -res > max_score:
                    # the move is better than the current best
                    max_score = -res
                    max_move = move
                
                # if there is a winning move now then directly return it
                if max_score == 1:
                    return max_move
            
            if incomplete:
                '''
                no winning move and not all moves are explored
                unknown outcome, leave it for later
                '''
                return None
            elif max_move and max_score > -1:
                '''
                no winning move and all moves are explored
                return the move that is at least better than
                losing
                '''
                return max_move
            else:
                '''
                all moves are losing moves
                leave it to simulations
                '''
                return None
        
        '''
        childrens are expanded already
        directly find the one with the same (dis)prove number
        '''
        pn, dn = root_node.get_numbers()
        if root_node.is_or_node:
            move, _ = root_node.find_equal_child_pn(root_node.children, pn)
        else:
            move, _ = root_node.find_equal_child_dn(root_node.children, dn)
        
        return move


