import os
from collections import deque
from copy import copy
from time import time
from typing import List

import dill
import jax
import jax.numpy as jnp
import numpy as np
from alphazero.edge import Edge
from alphazero.model import create_model, init_model
from env.macros import *
from jax.nn import softmax

from utils.test_utils import generate_random_game


def create_plane(inner_board: np.ndarray, player):
    assert inner_board.shape == (9, 9), f'inner board shape incompatible'
    assert player == X or player == O

    opponent = O if player == X else X
    first, second = np.zeros((9, 9)), np.zeros((9, 9))
    first[inner_board == player] = 1
    second[inner_board == opponent] = 1
    return np.stack((first, second))


def create_feature(history: deque, player):
    feature_list = []
    for state in history:
        inner_board = state['inner_board']
        feature_list.append(create_plane(inner_board, player))

    length = len(feature_list)
    if length < 8:
        empty_features = np.zeros(((8-length)*2, 9, 9))
        feature_list.append(empty_features)

    # make feature indicating the player
    player_feature = np.ones((1, 9, 9))*player
    feature_list.append(player_feature)

    feature = np.concatenate(feature_list)
    feature = np.expand_dims(feature, 0)

    assert feature.shape == (
        1, 17, 9, 9), f'feature has an erroneous shape of {feature.shape}'
    return feature


def get_val_and_pol(forward_func, feature: np.ndarray, valid_moves):
    feature = jnp.asarray(feature)
    val, logits = forward_func(feature)

    valid_logits = logits[0, valid_moves]
    probs = softmax(valid_logits)

    return val.item(), probs


def compute_puct_score(edge: Edge, total_visits, C):
    Q = edge.get_mean_action_val()
    P = edge.get_prior()
    N = edge.get_visit_count()
    UCB = C*P*np.sqrt(total_visits)/(1+N)

    return Q + UCB


def get_move_probs(moves: List[int], counts: np.ndarray, temperature: float):
    logits = counts/temperature
    probs = softmax(logits)
    prob_vector = np.zeros(81, dtype=np.float32)
    for move, prob in zip(moves, probs):
        prob_vector[move] = prob
    return prob_vector


def save_checkpoint(params, model_state, opt_state, replay_buffer, rand_key, train_steps: int, dir_path: str):
    checkpoint = {'params': params, 'model_state': model_state,
                  'opt_state': opt_state, 'replay_buffer': replay_buffer, 'rand_key': rand_key}

    ckpt_path = os.path.join(dir_path, f'checkpoints/{train_steps}.pickle')

    with open(ckpt_path, 'wb') as fp:
        dill.dump(checkpoint, fp)


def load_checkpoint(train_steps: int, dir_path: str):
    ckpt_path = os.path.join(dir_path, f'checkpoints/{train_steps}.pickle')
    with open(ckpt_path, 'rb') as fp:
        ckpt = dill.load(fp)
    params = ckpt['params']
    model_state = ckpt['model_state']
    opt_state = ckpt['opt_state']
    replay_buffer = ckpt['replay_buffer']
    rand_key = ckpt['rand_key']
    return params, model_state, opt_state, replay_buffer, rand_key


def main():
    # game1 = generate_random_game(5)
    # game2 = generate_random_game(6)
    # hist = deque([game1, game2])
    # feature = create_feature(hist, X)
    # model = create_model(True)

    # params, state = init_model(model)
    # val, probs = get_val_and_pol(model, params, state, feature, valid_moves=(0,1,2,3,5))
    # print(val)
    # print(probs)
    a = np.ones(3)
    print(type(softmax(a)))
    # a = jnp.zeros((2,9,9))
    # b = a[0]
    # print(b)


if __name__ == '__main__':
    main()
