import numpy as np
import player as ply
from termcolor import colored
from tqdm import tqdm
import time

from policy import RandomPolicy
class UltimateTTT:

    def __init__(self, player1, player2, state = None) -> None:
        if state is not None:
            self.inner_board = np.copy(state['inner'])
            self.outer_board = np.copy(state['outer'])
            self.current_player = state['current']
            self.game_end = state['game_end']
            self.winner = state['winner']
            self.previous_move = state['previous']
            self.next_valid_move = state['valid_move']
        else:
            # player x : 1, player o: -1, empty: 0
            self.inner_board = np.zeros((9,9))
            # player x: 1, player o: -1, incomplete: 0, tie: 2
            self.outer_board = np.zeros((3,3))

            # player x: 1, player o:-1
            self.current_player = 1

            self.game_end = False
            self.winner = None

            self.previous_move = None
            self.next_valid_move = list(range(0,81))

        self.player_x = player1
        self.player_o = player2


    '''
    return the current game state
    '''
    def get_state(self):
        state = {
            "inner":np.copy(self.inner_board),
            "outer":np.copy(self.outer_board),
            "current":self.current_player,
            "game_end":self.game_end,
            "winner":self.winner,
            "previous":self.previous_move,
            "valid_move":self.next_valid_move
        }
        return state
    

    '''
    given the move, update the game state
    '''
    def update_game_state(self,move):
        # update previous move
        self.previous_move = move
        # update board state
        self.update_board(move)
        # update next valid moves
        self.update_valid_moves()
        # check if there is a winner or draw
        self.check_winner()
        # switch player
        self.switch()


    '''
    The play function that implements the game loop.
    '''
    def play(self, display = False):
        while not self.game_end:
            if display:
                # print the board
                print("The inner board:\n")
                self.display_board(board='inner')
                print()
                print("The outer board:\n")
                self.display_board(board='outer')
                print()

            # check the player
            player = 'x' if self.current_player == 1 else 'o'
            
            if display:
                print(f"Player {player}'s turn to make a move:\n")         
                # display the valid next moves
                self.display_valid_moves()

            # make a move
            move = self.make_move()
            coordinate = self.map_coordinate(move,board='inner')

            if display:
                # print a line of separation
                self.print_line(color = True, length='very long')
            
                print(f'Player {player} played position {coordinate}.\n')

            # update game state
            self.update_game_state(move)
            
            
            
        if display:    
            print(colored("Game over!\n", 'green', attrs=['bold']))
            # display result
            if self.winner == 1:
                print(colored('The winner is player x!\n','green',attrs=['bold']))
            elif self.winner == -1:
                print(colored('The winner is player o!\n','green',attrs=['bold']))
            else:
                print(colored('The game is a draw.\n','green',attrs=['bold']))

            # print the board
            print("The inner board:\n")
            self.display_board(board='inner')
            print()
            print("The outer board:\n")
            self.display_board(board='outer')

    '''
    return the move chosen by the current player
    '''
    def make_move(self):
        current_state = self.get_state()
        if self.current_player == 1:
            candidate_move = self.player_x.move(current_state)
            while candidate_move not in self.next_valid_move:
                print('The move you specified is not valid. Try again!\n')
                candidate_move = self.player_x.move(current_state)
            
            return candidate_move

        else:
            candidate_move = self.player_o.move(current_state)
            while candidate_move not in self.next_valid_move:
                print('The move you specified is not valid. Try again!\n')
                candidate_move = self.player_o.move(current_state)
            
            return candidate_move
    
    '''
    take a 3x3 board as an argument, return the following:
    1 if player x has won this board
    -1 of player o has won this board
    2 if both players tie in this board
    0 if the board is not won and non-complete
    '''
    def check_board(self,board:np.array):
        for i in range(3):
            # check row for x
            if all(board[i]== 1):
                return 1
            # check row for o
            elif all(board[i]== -1):
                return -1
            # check column for x
            elif all(board[:,i]== 1):
                return 1
            # check column for o
            elif all(board[:,i]== -1):
                return -1
        
        # check diagonal for x
        if board[0,0] == 1 and board[1,1] == 1 and board[2,2] == 1:
            return 1
        elif board [0,2] == 1 and board[1,1] == 1 and board[2,0] == 1:
            return 1
        
        # check diagonal for o
        if board[0,0] == -1 and board[1,1] == -1 and board[2,2] == -1:
            return -1
        elif board [0,2] == -1 and board[1,1] == -1 and board[2,0] == -1:
            return -1
        
        # No one has won at this point

        # check incompelte
        if any(board[0] == 0) or any(board[1]==0) or any(board[2]==0):
            return 0
        # no slot left, must be tie
        else:
            return 2


    '''
    update the inner and outer board states based on the move
    '''
    def update_board(self, move):
        # update inner board value first
        inner_row, inner_col = self.map_coordinate(move, board='inner')
        self.inner_board[inner_row, inner_col] = self.current_player
        # update the outer board based on the sub-board that's updated
        outer_row, outer_col = self.map_coordinate(move, board='outer')
        sub_board = self.inner_board[outer_row*3:outer_row*3 + 3, outer_col*3:outer_col*3 + 3]
        self.outer_board[outer_row, outer_col] = self.check_board(sub_board)
    

    '''
    update the valid moves based on the previous move
    '''
    def update_valid_moves(self):
        valid_moves = []
        if self.previous_move is not None:
            # get coordinate in the sub-board
            sub_row, sub_col = self.map_coordinate(self.previous_move, board='sub')

            # if the outer board is incomplete, only the slots inside the sub-board are allowed
            if self.outer_board[sub_row, sub_col] == 0:
                for i in range(sub_row*3, sub_row*3 + 3):
                    for j in range(sub_col*3, sub_col*3 + 3):
                        if self.inner_board[i,j] == 0:
                            valid_moves.append(self.map_value(i,j))

            # otherwise the next player can choose to play any valid sub-board
            else:
                # loop through outer board
                for m in range(3):
                    for n in range(3):
                        # can play on the current sub-board
                        if self.outer_board[m,n] == 0:
                            for i in range(m*3, m*3 + 3):
                                for j in range(n*3, n*3 + 3):
                                    if self.inner_board[i,j] == 0:
                                        valid_moves.append(self.map_value(i,j))
            
            self.next_valid_move = valid_moves
            

    '''
    check if the game is won
    1 if x is the winner
    -1 if o is the winner
    2 for tie in the entire game
    0 for no winner yet
    '''
    def check_winner(self):
        res = self.check_board(self.outer_board)
        if res != 0:
            self.winner = res
            self.game_end = True
    
    '''
    switch the current player
    '''
    def switch(self):
       self. current_player = self.current_player*-1


    '''
    Helper function that maps move (0-80) to board coordinates
    '''
    def map_coordinate(self, move, board = 'inner'):
        row = move // 9
        col = move % 9
        if board == 'inner':
            return row, col
        elif board == 'outer':
            outer_row = row//3
            outer_col = col//3
            return outer_row, outer_col
        elif board == 'sub':
            sub_row = row % 3
            sub_col = col % 3
            return sub_row, sub_col
        else:
            raise ValueError('Board name not recognized.')


    '''
    Helper function that maps inner board coordinate to move value (0-80)
    '''
    def map_value(self, row, col):
        return row*9 + col

    
    '''
    Helper function to print a horizontal line
    color: swtich the color on and off
    length: very long for game turn separation, long for inner board, short for outer board
    '''
    def print_line(self, color = False, length = 'long'):
        if length == 'long':
            print("  ",end='')
            if color:
                print(colored(' ---'*9, 'yellow',attrs=['bold']))
            else:
                print(' ---'*9)
        elif length == 'short':
            print("  ",end='')
            if color:
                print(colored(' ---'*3, 'yellow',attrs=['bold']))
            else:
                print(' ---'*3)
        elif length == 'very long':
            if color:
                print(colored('='*100, 'red',attrs=['bold']))
            else:
                print('-'*30)
        else:
            raise ValueError('Line length not recognized.')

    '''
    display the valid moves in coordinates of the inner board
    '''
    def display_valid_moves(self):
        valid_move_coord = map(self.map_coordinate, self.next_valid_move, ['inner']*len(self.next_valid_move))
        valid_move_coord = map(str, list(valid_move_coord))
        valid_move_coord = ' '.join(valid_move_coord)
        print('valid positions: ' + valid_move_coord + '\n')
    '''
    display the selected board to the console
    '''
    def display_board(self, board = 'inner'):
        if board == 'inner':
            # print col number
            indices = map(str, list(range(9)))
            indices = "   ".join(indices)
            indices = "    "+ indices
            print(indices)

            for i, row in enumerate(self.inner_board):
                # print horizontal line
                if i % 3 == 0:
                    self.print_line(color=True)
                else:
                    self.print_line(color=False)
                
                # print row number
                print(i,end=" ")

                # print markers and vertical lines
                for j,item in enumerate(row):
                    print(colored('|', 'yellow',attrs=['bold']),end='') if j % 3 == 0 else print('|', end='')
                    if item == 1:
                        print(colored(' x ','cyan'), end='')
                    elif item == -1:
                        print(colored(' o ','magenta'), end='')
                    else:
                        print('   ', end='')
                print(colored('|', 'yellow',attrs=['bold']))

            # print bottom line
            self.print_line(color=True)

        elif board == 'outer':
            # print the col numbers
            indices = map(str, list(range(3)))
            indices = "   ".join(indices)
            indices = "    "+ indices
            print(indices)

            for i, row in enumerate(self.outer_board):
                # print horizontal line
                self.print_line(color=True, length= 'short')
                
                # print the row numbers
                print(i,end=" ")

                # print markers and vertical lines
                for j,item in enumerate(row):
                    print(colored('|', 'yellow',attrs=['bold']),end='')
                    if item == 1:
                        print(colored(' x ','cyan'), end='')
                    elif item == -1:
                        print(colored(' o ','magenta'), end='')
                    else:
                        print('   ', end='')
                print(colored('|', 'yellow',attrs=['bold']))

            # print bottom line
            self.print_line(color=True, length='short')
        else:
            raise ValueError('Board name not recognized.')
    
    

def main():
    pol = RandomPolicy()
    sim = ply.RandomPlayer()
    p1 = ply.AlphaZeroPlayer()
    p2 = ply.MCTSPlayer(pol, sim)

   
    x_win = 0
    o_win = 0
    tie = 0

    for _ in tqdm(range(10)):
        game = UltimateTTT(player1=p1, player2=p2)
        game.play(False)
        final_state = game.get_state()
        if final_state['winner'] == 1:
            x_win+=1
        elif final_state['winner'] == -1:
            o_win+=1
        elif final_state['winner'] == 2:
            tie += 1
        
        p1.reset()
        p2.reset()
        
        
    
    print('number of wins for x:', x_win)
    print('number of wins for o:', o_win)
    print('number of ties:', tie)
    
    
  
    # game.display_board()
if __name__ == '__main__':
    main()


