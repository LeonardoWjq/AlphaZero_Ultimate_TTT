import os
import pickle
from typing import List

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import yaml
from env.macros import *
from env.ultimate_ttt import UltimateTTT
from jax import jit, value_and_grad
from tqdm import tqdm

from alphazero.core import AlphaZero
from alphazero.model import create_model, init_model
from alphazero.record import Record
from alphazero.replay_buffer import ReplayBuffer

dir_path = os.path.dirname(__file__)


def self_play(model_params: dict, model_state: dict, PRNGkey, sim_num: int, explore_factor: float, temperature: float):
    px = AlphaZero(model_params, model_state, PRNGkey,
                   sim_num, explore_factor, temperature)
    po = AlphaZero(model_params, model_state, PRNGkey,
                   sim_num, explore_factor, temperature)
    game = UltimateTTT(None, None)
    game_state = game.get_state()

    while game_state['outcome'] == INCOMPLETE:
        current_player = game_state['current_player']
        if current_player == X:
            move = px.get_move(game_state)
        else:
            move = po.get_move(game_state)

        game.update_state(move)
        game_state = game.get_state()

    px_trajectory: List[Record] = px.get_traj_record()
    po_trajectory: List[Record] = po.get_traj_record()

    if game_state['outcome'] == TIE:
        for record in px_trajectory:
            record.set_score(0.0)
        for record in po_trajectory:
            record.set_score(0.0)
    elif game_state['outcome'] == X_WIN:
        for record in px_trajectory:
            record.set_score(1.0)
        for record in po_trajectory:
            record.set_score(-1.0)
    else:
        for record in px_trajectory:
            record.set_score(-1.0)
        for record in po_trajectory:
            record.set_score(1.0)

    return px_trajectory, po_trajectory


def train(total_games: int, games_per_train: int, iterations: int, lr: float, batch_size, seed: int, self_play_args: dict):
    model = create_model(True)
    params, model_state = init_model(model)
    opt = optax.adamw(lr)
    opt_state = opt.init(params)
    rand_gen = hk.PRNGSequence(seed)
    replay_buffer = ReplayBuffer(seed)

    @jit
    def loss_func(params, state, feature, true_score, search_prob):
        (pred_score, logits), next_state = model.apply(params, state, feature)
        val_loss = optax.l2_loss(pred_score, true_score).mean()
        pol_loss = optax.softmax_cross_entropy(logits, search_prob).mean()
        overall_loss = val_loss + pol_loss
        return overall_loss, next_state

    grad_func = jit(value_and_grad(loss_func, has_aux=True))

    @jit
    def step(feature, true_score, search_prob):
        (total_loss, next_model_state), gradient = grad_func(
            params, model_state, feature, true_score, search_prob)
        update, next_opt_state = opt.update(gradient, opt_state, params)
        next_params = optax.apply_updates(params, update)
        return next_params, next_model_state, total_loss, next_opt_state

    for index in tqdm(range(total_games)):
        next_key = rand_gen.next()
        px_traj, po_traj = self_play(
            model_params=params, model_state=model_state, PRNGkey=next_key, **self_play_args)
        replay_buffer.add_traj(px_traj)
        replay_buffer.add_traj(po_traj)

        if (index + 1) % games_per_train == 0:
            for _ in range(iterations):
                feature, search_prob, true_score = replay_buffer.sample_data(
                    batch_size)
                next_params, next_model_state, total_loss, next_opt_state = step(
                    feature, true_score, search_prob)
                params = next_params
                model_state = next_model_state
                opt_state = next_opt_state
                print(total_loss)


def main():
    with open(os.path.join(dir_path, 'config.yaml'), 'r') as fp:
        config = yaml.safe_load(fp)
    train(**config)


if __name__ == '__main__':
    main()
