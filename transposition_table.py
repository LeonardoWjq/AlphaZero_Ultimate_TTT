from environment import UltimateTTT
from player import RandomPlayer
from collections import namedtuple
from termcolor import colored
import time

# A handy data type to hold record of the game state
Record = namedtuple('Record',['state', 'value', 'move'])

class Transposition:
    
    def __init__(self, player1, player2, root_player = 1, num_rand = 0) -> None:
        self.p1 = player1
        self.p2 = player2
        self.num_rand = num_rand
        self.game = UltimateTTT(self.p1, self.p2)
        self.root_player = root_player
        self.trans_table = {} # transposition table as a dictionary 
    
    def random_play(self):
        for _ in range(self.num_rand):
            if self.game.game_end:
                break
            move = self.game.make_move()
            self.game.update_game_state(move)
    
    '''
    generate hash code of a game state based on the inner board
    '''
    def hash_function(self, state):
        inner_board = state['inner']
        key = ''
        for row in inner_board:
            for entry in row:
                key += str(entry)
        
        return key
    
    '''
    check if two states are identical
    '''
    def equal_state(self, first, second):
        if  (first['inner'] != second['inner']).any():
            return False
        
        if first['current'] != second['current']:
            return False

        if first['winner'] != second['winner']:
            return False
        
        if first['previous'] != second['previous']:
            return False
        
        return True


    def alpha_beta_tt(self, state, alpha, beta) -> int:
        # statically evaluate
        if state['game_end']:
            # if it is a tie
            if state['winner'] == 2:
                return 0
            # current player winning
            if state['winner'] == state['current']:
                return 1
            # current player losing
            else:
                return -1
        else:
            retrieved_record = None
            hash_code = self.hash_function(state)

            # found entry in transposition table
            if hash_code in self.trans_table:
                record_list = self.trans_table[hash_code]
                for record in record_list:
                    # found the exact record
                    if self.equal_state(record.state, state):
                        retrieved_record = record
                        break

            if retrieved_record is not None:
                value = retrieved_record.value
                if value >= beta:
                    return beta
                elif value < alpha:
                    return alpha
                else:
                    return value
            
            # Did not find exact record

            legal_moves = state['valid_move']
            game = UltimateTTT(self.p1, self.p2, state)
            best_move = -1
            best_value = -100
            for move in legal_moves:
                game.update_game_state(move)            
                value = - self.alpha_beta_tt(game.get_state(), -beta, -alpha)
                game.undo()
                # update best to be alpha
                alpha = max(alpha, value)
                if value > best_value:
                    best_value = value
                    best_move = move

                # beta cut
                if (value >= beta):                    
                    return beta

            # get exact value now
            if hash_code in self.trans_table:
                self.trans_table[hash_code].append(Record(state, best_value, best_move))
            else:
                self.trans_table[hash_code] = [Record(state, best_value, best_move)]

            return alpha
                


    def run(self):
        self.random_play()
        self.game.display_board()
        self.game.display_board(board='outer')
        

        start_time = time.time()
        # get the result for the current player
        value = self.alpha_beta_tt(self.game.get_state(),float('-inf'), float('inf'))
        print(f'Time to run Alpha-Beta Search: {time.time() - start_time}')

        return value
    
    def demonstrate_moves(self) -> None:
        while not self.game.game_end:
            curr_state = self.game.get_state()
            hash_code = self.hash_function(curr_state)
            record_list = self.trans_table[hash_code]
            move = None
            for record in record_list:
                if self.equal_state(curr_state,record.state):
                    move = record.move
                    break
            assert move is not None
            row = move // 9
            col = move % 9
            player = 'x' if self.game.current_player == 1 else 'o'
            print(colored(f'Player {player} made a move {(row,col)}.','green'))
            self.game.update_game_state(move)
            self.game.display_board('inner')
            self.game.display_board('outer')

    
def main():
    player1 = RandomPlayer()
    player2 = RandomPlayer()
    agent = Transposition(player1, player2, num_rand=50)
    result = agent.run()
    agent.demonstrate_moves()
    current_player = 'x' if agent.game.current_player == 1 else 'o'
    print(f'Value of the current player {current_player} is {result}.')

if __name__ == '__main__':
    main()