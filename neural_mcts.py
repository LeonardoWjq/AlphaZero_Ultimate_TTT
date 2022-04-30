import environment as env
import numpy as np
import torch
from neural_net import Network
from environment import UltimateTTT as game


# The class for one TreeNode
class NeuralTreeNode:
    '''
    game_state: the current game state the tree node stores
    simulation_player: the player that is used in simulation step
    '''
    def __init__(self,game_state:dict,simulation_player,depth:int):
        self.state = game_state
        self.sim_player = simulation_player
        self.depth = depth
        self.edges = {}
        for move in game_state['valid_move']:
            self.edges[move] = {'count':0, 'total_val':0, 'node':None}
        self.is_terminal = self.state['game_end']
    
    '''
    unroll from the node till the end of the game
    return the game winner
    x win: 1
    o win: -1
    draw: 2
    '''
    def unroll(self):
        if self.is_terminal:
            return self.state['winner']
        game = env.UltimateTTT(self.sim_player, self.sim_player,state = self.state)
        game.play()
        final_state = game.get_state()
        return final_state['winner']

    '''
    get a list of actions that have the max values
    exploration_factor: controls the level of exploration, the higher, the more exploration and vice-versa
    prioritize the nodes that have not been expanded before
    randomly sample one action from the candidates in case of a tie
    return the sampled acton
    '''
    def get_max_action(self,explore_factor):
        # get total visit of this node
        total_visit = 0
        for _, value in self.edges.items():
            total_visit += value['count']
            
        # store maximum value
        max_val = float('-inf')
        # store max actions
        max_actions = []
        for action, record in self.edges.items():
            # if the edge is never visited before
            if record['count'] == 0:
                # if the previous are visited edges
                if max_val < float('inf'):
                    max_val = float('inf')
                    max_actions = [action]
                else:
                    max_actions.append(action)
            # if the edge if visited before
            else:
                q = record['total_val']/record['count']
                u = np.sqrt(np.log(total_visit)/record['count'])
                val = q + explore_factor*u

                # if the value is the current max
                if val > max_val:
                    max_val = val
                    max_actions = [action]
                # if it is equal the current max
                elif val == max_val:
                    max_actions.append(action)
        
        # choose action to simulate
        action = np.random.choice(max_actions)

        return action

    '''
    simulate and (possibly) grow the tree once
    return the simulation result
    x win: 1
    o win: -1
    tie: 2 
    '''
    def simulate(self, explore_factor):
        # check if current is terminal state
        if self.is_terminal:
            return self.state['winner']
        
        # get the max action
        action = self.get_max_action(explore_factor)

        # get the edge of that action
        edge = self.edges[action]

        # grow the edge if it does not exist
        if edge['node'] is None:
            # get the updated state
            game = env.UltimateTTT(self.sim_player,self.sim_player,self.state)
            game.update_game_state(action)
            new_state = game.get_state()

            # create new tree node
            new_node = NeuralTreeNode(new_state,self.sim_player,self.depth+1)
            edge['node'] = new_node
            
            # unroll the new_node once
            result = new_node.unroll()

            # initialize counter
            edge['count'] = 1

            # update value
            # if result is not tie
            if result != 2:
                # initialize value using the result
                # positive if same sign, negative otherwise
                edge['total_val'] = result*self.state['current']
            # return the simulation result
            return result
        # simulate next node if it exists
        else:
            result = edge['node'].simulate(explore_factor)
            edge['count'] += 1
            if result != 2:
                # increment total_val based on the result
                edge['total_val'] += result*self.state['current']
            return result
    
    '''
    input a state
    check if the state is identical to the state of itself
    '''
    def equal_state(self, state:dict):
        
        this_state = self.state
        input_state = state

        # if the inner boards have any difference
        if (this_state['inner'] != input_state['inner']).any():
            return False
               
        # if the current players are not the same
        if this_state['current'] != input_state['current']:
            return False
        
        
        # if the winners are not the same
        if this_state['winner'] != input_state['winner']:
            return False
        
        # if the previous moves are not the same
        if this_state['previous'] != input_state['previous']:
            return False
        
        # the two states are identical
        return True
    
    '''
    return the depth of the node
    '''
    def get_depth(self)->int:
        return self.depth
    
    '''
    return the game state
    '''
    def get_state(self)->dict:
        return self.state





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Monte Carlo Tree Search Class
class NeuralMCTS:
    
    # initialze attributes
    def __init__(self,state:dict,simulation_player,exploration_factor=0.5,threshold=65,is_regression=True) -> None:
        self.root = NeuralTreeNode(state,simulation_player,self.count_step(state['inner']))
        self.sim_player = simulation_player
        self.explore_factor = exploration_factor
        self.is_regression = is_regression
        # threshold to use neural network for action selection instead
        self.threshold = threshold
        if is_regression:
            self.network = Network(True)
            self.network.load_state_dict(torch.load('regression_model.pt'))
            self.network.to(device)
            self.network.eval()
        else:
            self.network = Network(False)
            self.network.load_state_dict(torch.load('classification_model.pt'))
            self.network.to(device)
            self.network.eval()


    '''
    input number of simulation steps
    run simulations from the root (current) state
    '''
    def run_simumation(self, num = 600):
        if self.root.get_depth()<self.threshold:
            for _ in range(num):
                self.root.simulate(self.explore_factor)


    '''
    compute the probability distributions for candidate moves
    sample an action from the distribution
    return the next action
    ATTENTION: this method also changes the root node to the lastest game state resulted after taking the action provided it's not None
    '''
    def get_move(self):
        depth = self.root.get_depth()
        if depth < self.threshold:
            actions = []
            visit_count = []
            # store the actions and the corresponding visit counts
            for key, value in self.root.edges.items():
                actions.append(key)
                visit_count.append(value['count'])
            
            visit_count = np.array(visit_count)
            
            # get the action probabilities
            probs = visit_count/np.sum(visit_count)

            # sample next move
            next_move = np.random.choice(actions, p = probs)

            # transplant to the next node
            next_node = self.root.edges[next_move]['node']
            if next_node is not None:
                self.root = next_node

            return next_move,'simulated'
        else:
            game_state = self.root.get_state()
            next_moves, outputs = self.inference(game_state)
            if self.is_regression:
                # minimum of opponent is max move to the current player
                max_move_index = torch.argmin(outputs, dim=0).item()
                return next_moves[max_move_index],'inferred'
            else:
                
                preds = torch.argmax(outputs,dim=1)
                # The lower the chance the opponent will win
                # the higher the chance the current player will win
                max_class = -1
                max_actions = []
                for index,pred in enumerate(preds):
                    if pred.item() > max_class:
                        max_class = pred.item()
                        max_actions = [next_moves[index]]
                    elif pred.item() == max_class:
                        max_actions.append(next_moves[index])
                # random break tie
                next_move = np.random.choice(max_actions)
                return next_move,'inferred'




    def inference(self, state:dict):
        # list of (next move, next inner board, next outer board, next valid moves, next current player)
        next_moves = []
        next_inners = []
        next_outers = []

        def legal_move_repr(legal_moves):
            feature_map = np.zeros((9,9))
            for move in legal_moves:
                row = move//9
                col = move%9
                feature_map[row,col] = 1
            return feature_map

        for move in state['valid_move']:
            env = game(None,None,state,False)
            env.update_game_state(move)
            new_state = env.get_state()

            new_current_player = new_state['current']
            new_inner_board = new_state['inner']*new_current_player
            new_outer_board = new_state['outer']*new_current_player
            new_valid_moves = new_state['valid_move']

            new_move_feature = legal_move_repr(new_valid_moves)
            new_inner_board = np.concatenate((new_inner_board[None], new_move_feature[None]))

            next_moves.append(move)
            next_inners.append(new_inner_board)
            next_outers.append(new_outer_board)
        
        next_inners = np.array(next_inners)
        next_outers = np.array(next_outers)
        next_inners = torch.tensor(next_inners, dtype=torch.float64).to(device)
        next_outers = torch.tensor(next_outers, dtype=torch.float64).view(-1,9).to(device)

        output = self.network(next_inners, next_outers)
        
        return next_moves, output


    
    '''
    update the root node based on the input state:
    if the state is already simulated by the tree before, then just change the root node to the corresponding node
    otherwise, create a new node and make it the root (typically happens when there is a new game)
    '''
    def transplant(self, state:dict):
        # get the previous move for faster locationing
        prev_move = state['previous']

        if prev_move is not None:
            target_node = self.root.edges[prev_move]['node']
            if target_node is not None and target_node.equal_state(state):
                self.root = target_node
                return
        
        # either the previous move is None or the target node is not matching
        new_node = NeuralTreeNode(state,self.sim_player,self.count_step(state['inner']))
        self.root = new_node
    
    def count_step(self, board:np.array)->int:
        '''
        count the number of pieces into the game
        '''
        total = 0
        for row in board:
            for piece in row:
                if piece != 0:
                    total += 1
        
        return total