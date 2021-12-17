import player as ply
import environment as env
import numpy as np

# The class for one TreeNode
class TreeNode:
    '''
    game_state: the current game state the tree node stores
    policy: a function that maps game states to action probabilities, used as prior
    simulation_player: the player that is used in simulation step
    '''
    def __init__(self,game_state: dict, policy, simulation_player):
        self.state = game_state
        self.policy = policy
        self.sim_player = simulation_player
        self.priors = self.policy.get_probs(game_state)
        self.edges = {}
        for i, move in enumerate(game_state['valid_move']):
            self.edges[move] = {'prior':self.priors[i], 'count':0, 'total_val':0, 'node':None}
        
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
                u = explore_factor*record['prior']*np.sqrt(total_visit)/(1+record['count'])
                # if the value is the current max
                if q+u > max_val:
                    max_val = q+u
                    max_actions = [action]
                # if it is equal the current max
                elif q+u == max_val:
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
            new_node = TreeNode(new_state, self.policy, self.sim_player)
            edge['node'] = new_node
            
            # unroll the new_node once
            result = new_node.unroll()

            # initialize counter
            edge['count'] = 1

            # update value
            # if result is not tie
            if result != 2:
                current_player = self.state['current']
                # initialize value using the result
                # positive if same sign, negative otherwise
                edge['total_val'] = result*current_player
            # return the simulation result
            return result
        # simulate next node if it exists
        else:
            result = edge['node'].simulate(explore_factor)
            edge['count'] += 1
            if result != 2:
                current_player = self.state['current']
                # increment total_val based on the result
                edge['total_val'] += result*current_player
            
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
        
        # if the outer boards have any difference
        if (this_state['outer'] != input_state['outer']).any():
            return False
        
        # if the current players are not the same
        if this_state['current'] != input_state['current']:
            return False
        
        # if the game end signals are not the same
        if this_state['game_end'] != input_state['game_end']:
            return False
        
        # if the winners are not the same
        if this_state['winner'] != input_state['winner']:
            return False
        
        # if the previous moves are not the same
        if this_state['previous'] != input_state['previous']:
            return False
        
        # if the valid moves are not the same
        if this_state['valid_move'].sort() != input_state['valid_move'].sort():
            return False
        
        # the two states are identical
        return True






# Monte Carlo Tree Search Class
class MCTS:
    # initialze attributes
    def __init__(self, state:dict, policy, simulation_player, exploration_factor = 0.7) -> None:
        self.root = TreeNode(state, policy, simulation_player)
        self.pol = policy
        self.sim_player = simulation_player
        self.explore_factor = exploration_factor

    '''
    input number of simulation steps
    run simulations from the root (current) state
    '''
    def run_simumation(self, num = 500):
        for _ in range(num):
            self.root.simulate(self.explore_factor)

    '''
    compute the probability distributions for candidate moves
    sample an action from the distribution
    return both the action and the probability disitribution of moves over the entire output space (81 slots)
    ATTENTION: this method also changes the root node to the lastest game state resulted after taking the action provided it's not None
    '''
    def get_move(self, temp):
        actions = []
        visit_count = []
        # store the actions and the corresponding visit counts
        for key, value in self.root.edges.items():
            actions.append(key)
            visit_count.append(value['count'])
        
        visit_count = np.array(visit_count)
        scores = visit_count**(1/temp)
        # get the action probabilities
        probs = scores/np.sum(scores)

        # sample next move
        next_move = np.random.choice(actions, p = probs)

        # the probabilities over the entire output dimension
        prob_vec = np.zeros(81)
        prob_vec[actions] = probs

        # transplant to the next node
        next_node = self.root.edges[next_move]['node']
        if next_node is not None:
            self.root = next_node

        return next_move, prob_vec
    
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
        new_node = TreeNode(state,self.pol,self.sim_player)
        self.root = new_node


        
    

        
        



# def main():
#     p1 = ply.RandomPlayer()
#     p2 = ply.RandomPlayer()
#     game = env.UltimateTTT(player1=p1, player2=p2)
#     pol = RandomPolicy()
#     x_win = 0
#     o_win = 0
#     tie = 0
#     # for _ in range(1000):
#     #     res = node.unroll()
#     #     if res == 1:
#     #         x_win += 1
#     #     elif res == -1:
#     #         o_win += 1
#     #     elif res == 2:
#     #         tie += 1
#     # print(x_win)
#     # print(o_win)
#     # print(tie)


# if __name__ == '__main__':
#     main()
   
    
        
        

        
