from environment import UltimateTTT
from collections import namedtuple
from termcolor import colored
import time

# A handy data type to hold record of the game state
Record = namedtuple('Record',['state', 'value', 'move'])

class Alpha_Beta_TT:
    
    def __init__(self, game:UltimateTTT, target_player = 1) -> None:
        self.game = game
        self.root = game.current_player
        self.target = target_player
        self.trans_table = {} # transposition table as a dictionary 
    
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


    def alpha_beta_tt(self, alpha, beta) -> int:
        # statically evaluate
        if self.game.game_end:
            # if it is a tie
            if self.game.winner == 2:
                return 0
            # current player winning
            if self.game.winner == self.game.current_player:
                return 1
            # current player losing
            else:
                return -1
        else:
            retrieved_record = None
            current_state = self.game.get_state()
            hash_code = self.hash_function(current_state)

            # found entry in transposition table
            if hash_code in self.trans_table:
                record_list = self.trans_table[hash_code]
                for record in record_list:
                    # found the exact record
                    if self.equal_state(record.state, current_state):
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

            legal_moves = self.game.next_valid_move
            best_move = -1
            best_value = -100
            for move in legal_moves:
                self.game.update_game_state(move)            
                value = - self.alpha_beta_tt(-beta, -alpha)
                self.game.undo()
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
                self.trans_table[hash_code].append(Record(current_state, best_value, best_move))
            else:
                self.trans_table[hash_code] = [Record(current_state, best_value, best_move)]

            return alpha

    def run(self, display=False):
        if display:
            self.game.display_board()
            self.game.display_board(board='outer')
        
        start_time = time.time()
        # get the result for the current player
        value = self.alpha_beta_tt(-1, 1)
        time_used = time.time() - start_time
        print(f'Time to run Alpha-Beta Search with Transposition Table: {time_used}')

        res = None
        if value == 0:
            res = 0
        elif value == 1:
            res =  1 if self.root == self.target else -1
        else:
            res = -1 if self.root == self.target else 1

        return res, time_used
    
    def print_trace(self) -> None:
        num_moves = 0
        while not self.game.game_end:
            curr_state = self.game.get_state()
            hash_code = self.hash_function(curr_state)
            record_list = self.trans_table[hash_code]
            # find the best move from the table
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
            # update game state
            self.game.update_game_state(move)
            self.game.display_board('inner')
            self.game.display_board('outer')
            num_moves += 1

        # undo all the moves
        for _ in num_moves:
            self.game.undo()