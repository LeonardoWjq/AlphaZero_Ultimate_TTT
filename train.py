from collections import namedtuple
from Network import Network
from environment import UltimateTTT
from player import MCTSPlayer, NNPlayer, RandomPlayer
from policy import NNPolicy, RandomPolicy
from tqdm import tqdm
from termcolor import colored
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

'''
given a history list of (board, probs)
output sample and label sets in mini-batches
'''
def to_dataset(history, mini_size = 20):
    # shuffle
    random.shuffle(history)
    board_batch = []
    prob_batch = []
    for board, prob in history:
        board_batch.append(board[None])
        prob_batch.append(prob)
    
    board_batch = np.array(board_batch)
    prob_batch = np.array(prob_batch)

    board_batch = torch.from_numpy(board_batch)
    prob_batch = torch.from_numpy(prob_batch)

    mini_board = to_mini_batch(board_batch, mini_size)
    mini_prob = to_mini_batch(prob_batch, mini_size)

    return mini_board, mini_prob, board_batch, prob_batch

'''
Given a batch dataset and a size
Output a list of mini-batches specified by the mini-batch size
'''
def to_mini_batch(dataset, mini_size):
    batch = []
    dim = dataset.size()[0]
    for index in range(0,dim,mini_size):
        batch.append(dataset[index:index+mini_size])
    
    return batch




'''
let the player play game with itself
input a player and its copy
output the dataset generated by the game
'''
def self_play(player1:MCTSPlayer,player2:MCTSPlayer):
    game = UltimateTTT(player1,player2)
    game.play()
    hist1 = player1.get_history()
    hist2 = player2.get_history()
    data = hist1 + hist2
    player1.reset()
    player2.reset()
    return data

'''
cross-entropy loss of move probabilities
'''
def loss_function(pi, p):
    cel = nn.CrossEntropyLoss()
    pol_loss = cel(pi,p)
    
    return pol_loss

'''
Given a current best player, a baseline player and the number of games to play
Output a normalized score in [-1,1] for the curent best player based on the game results
'''
def eval(current_best:MCTSPlayer, baseline:MCTSPlayer, num_games = 20):
    score = 0
    print("Evaluation in progress:")
    for i in tqdm(range(num_games)):
        # alternating x and o
        if i % 2 == 0:
            game = UltimateTTT(current_best, baseline)
            game.play()
            final_state = game.get_state()
            if final_state['winner'] == 1:
                score +=1
            elif final_state['winner'] == -1:
                score -= 1
        else:
            game = UltimateTTT(baseline, current_best)
            game.play()
            final_state = game.get_state()
            if final_state['winner'] == -1:
                score +=1
            elif final_state['winner'] == 1:
                score -= 1
        
        # reset both players
        current_best.reset()
        baseline.reset()
    
    return score/num_games

def generate_dataset(num_self_play=100, checkpoint = 5):
    pol = RandomPolicy()
    sim = RandomPlayer()
    player = MCTSPlayer(pol, sim, store_hist=True)
    player_cpy = MCTSPlayer(pol, sim, store_hist=True)
    dataset = None
    try:
        with open('dataset.txt','rb') as fp:
            dataset = pickle.load(fp)
        print(f"Dataset file loaded. Currently has {len(dataset)} data points. Continue generating data.")
    except FileNotFoundError:
        dataset = []
        print('Cannot find dataset file. Start from the beginning.')
    
    for game_num in tqdm(range(num_self_play)):
        if game_num % checkpoint == 0:
            with open('dataset.txt', 'wb') as fp:
                pickle.dump(dataset,fp)
        
        data = self_play(player, player_cpy)
        dataset.extend(data)
    
    with open('dataset.txt', 'wb') as fp:
        pickle.dump(dataset,fp)


            

'''
num_self_play: number of self-playing games
num_epoch: number of epoch for each batch of data
mini_size: size of mini-batch
lr: learning rate
checkpoint: number of runs per save
start: starting number of model to continue
'''
def train(num_epoch = 30, mini_size = 20, lr = 1e-3, load_model = False):

    model = None
    total_loss = None

    try:
        with open('dataset.txt','rb') as fp:
            dataset = pickle.load(fp)
            print(colored('Dataset successfully loaded.','green'))
    except FileNotFoundError:
        print(colored('Cannot find dataset. Training aborted.','red'))
        return

    # check if the starting point is specified
    if load_model:
        model = torch.load('model.pt')
        print(colored('Model successfully loaded.', 'green'))
        with open('loss.txt','rb') as fp:
            total_loss = pickle.load(fp)
        print(colored('Loss record successfully loaded.','green'))
    else:
        print(colored('Training the network from fresh.','blue'))
        model = Network()
        total_loss = []

    optimizer = optim.Adam(model.parameters(),lr)

    print(colored('Start training process:', 'green'))

    # get batch and split into mini-batches    
    mini_board, mini_prob, board_batch, prob_batch = to_dataset(dataset, mini_size)

    
    # training the network
    for epoch in tqdm(range(num_epoch)):
        for index in range(len(mini_board)):
            board = mini_board[index]
            pi = mini_prob[index]

            p = model(board)
            optimizer.zero_grad()
            loss = loss_function(pi,p)
            loss.backward()
            optimizer.step()
        
        batch_p = model(board_batch)
        total_loss.append(loss_function(prob_batch, batch_p).item())
        
    

    
    # saving the model and the loss record at the end
    with open('loss.txt', 'wb') as fp:
        pickle.dump(total_loss,fp)    
    torch.save(model,'model.pt')

def plot_figure():
    with open('loss.txt','rb') as fp:
        loss = pickle.load(fp)
        plt.plot(range(1,len(loss)+1), loss)
        plt.show()


def main():
    # generate_dataset()
    train(num_epoch=100, lr=0.0005,load_model=False, mini_size=10)
    plot_figure()
    
    


if __name__ == '__main__':
    main()