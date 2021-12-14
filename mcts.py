import player as ply
import environment as env
from policy import RandomPolicy
import numpy as np
# The class for one TreeNode


class TreeNode:
    '''
    parent_node: the parent tree node object
    game_state: the current game state the tree node stores
    policy: a function that maps game states to action probabilities, used as prior
    simulation_player: the player that is used in simulation step
    '''
    def __init__(self, parent_node, game_state: dict, policy, simulation_player):
        self.parent = parent_node
        self.state = game_state
        self.policy = policy
        self.sim_player = simulation_player
        self.priors = self.policy.get_probs(game_state)
        self.edges = {}
        for i, move in enumerate(game_state['valid_move']):
            self.edges[move] = {'prior':self.priors[i], 'count':0, 'total_val':0, 'node':None}
        
        self.is_terminal = self.state['game_end']
    
    def unroll(self):
        if self.is_terminal:
            return self.state['winner']
        game = env.UltimateTTT(self.sim_player, self.sim_player,state = self.state)
        game.play()
        final_state = game.get_state()
        return final_state['winner']

    '''
    simulate once
    return the game result 
    '''
    def simulate(self):
        # check if current is terminal state
        if self.is_terminal:
            return self.state['winner']
        
        max_val = float('-inf')
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
                u = record['prior']/(1+record['count'])
                # if the value is the current max
                if q+u > max_val:
                    max_val = q+u
                    max_actions = [action]
                # if it is equal the current max
                elif q+u == max_val:
                    max_actions.append(action)
        
        # choose action to simulate
        action = np.random.choice(max_actions)
        # get the edge of that action
        edge = self.edges[action]

        
        # grow the edge if it does not exist
        if edge['node'] is None:
            # get the updated state
            game = env.UltimateTTT(self.sim_player,self.sim_player,self.state)
            game.update_game_state(action)
            new_state = game.get_state()

            # create new tree node
            new_node = TreeNode(self, new_state, self.policy, self.sim_player)
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
            result = edge['node'].simulate()
            edge['count'] += 1
            if result != 2:
                current_player = self.state['current']
                # increment total_val based on the result
                edge['total_val'] += result*current_player
            
            return result

class MCTS:
    def __init__(self, state:dict, policy, simulation_player) -> None:
        self.root = TreeNode(None, state, policy, simulation_player)
    
    def run_simumation(self, num = 2000):
        for _ in range(num):
            self.root.simulate()
    
    def get_move(self):
        actions = []
        visit_count = []
        for key, value in self.root.edges.items():
            actions.append(key)
            visit_count.append(value['count'])
        
        visit_count = np.array(visit_count)
        probability = visit_count/np.sum(visit_count)
        next_move = np.random.choice(actions, p = probability)
        return next_move


        
    

        
        



# def main():
#     p1 = ply.RandomPlayer()
#     p2 = ply.RandomPlayer()
#     game = env.UltimateTTT(player1=p1, player2=p2)
#     policy = RandomPolicy()
#     mcts = MCTS(game.get_state(), policy=policy, simulation_player=p1)

#     start = time.time()
#     mcts.run_simumation(1000)
#     print(mcts.get_move())

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
   
    
        
        

        
