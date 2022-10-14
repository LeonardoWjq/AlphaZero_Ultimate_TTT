import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import dill
import optax
import yaml
from env.macros import *
from env.ultimate_ttt import UltimateTTT
from jax import grad, jit, random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from alphazero.core import AlphaZero
from alphazero.model import create_model, init_model
from alphazero.record import Record
from alphazero.replay_buffer import ReplayBuffer

dir_path = os.path.dirname(__file__)


def self_play(model_params: dict, model_state: dict, PRNGkey, sim_num: int, explore_factor: float, temperature: float, alpha: float, epsilon: float):
    px = AlphaZero(model_params, model_state, PRNGkey,
                   sim_num, explore_factor, temperature, alpha, epsilon)
    po = AlphaZero(model_params, model_state, PRNGkey,
                   sim_num, explore_factor, temperature, alpha, epsilon)
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


def train(total_games: int, games_per_train: int, iters_per_train: int, ckpt_frequency: int, lr: float, batch_size, seed: int, self_play_args: dict, last_ckpt: int = 0):
    assert last_ckpt >= 0
    model = create_model(True)
    opt = optax.adamw(lr)
    train_steps = last_ckpt
    writer = SummaryWriter(os.path.join(dir_path, 'log'))

    if last_ckpt > 0:
        ckpt_path = os.path.join(dir_path, f'checkpoints/{train_steps}.pickle')
        with open(ckpt_path, 'rb') as fp:
            ckpt = dill.load(fp)
        params = ckpt['params']
        model_state = ckpt['model_state']
        opt_state = ckpt['opt_state']
        replay_buffer = ckpt['replay_buffer']
        rand_key = ckpt['rand_key']
    else:
        params, model_state = init_model(model)
        opt_state = opt.init(params)
        replay_buffer = ReplayBuffer(seed)
        rand_key = random.PRNGKey(seed)

    @jit
    def loss_func(params, state, feature, true_score, search_prob):
        (pred_score, logits), next_state = model.apply(params, state, feature)
        val_loss = optax.l2_loss(pred_score, true_score).mean()
        pol_loss = optax.softmax_cross_entropy(logits, search_prob).mean()
        overall_loss = val_loss + pol_loss
        return overall_loss, (next_state, val_loss, pol_loss)

    grad_func = jit(grad(loss_func, has_aux=True))

    @jit
    def step(feature, true_score, search_prob):
        gradient, (next_model_state, val_loss, pol_loss) = grad_func(
            params, model_state, feature, true_score, search_prob)
        update, next_opt_state = opt.update(gradient, opt_state, params)
        next_params = optax.apply_updates(params, update)
        return next_params, next_model_state, next_opt_state, val_loss, pol_loss

    for index in tqdm(range(0, total_games, games_per_train)):
        with ThreadPoolExecutor(max_workers=4) as executor:
            res = []
            for _ in range(games_per_train):
                rand_key, next_key = random.split(rand_key)
                res.append(executor.submit(self_play, model_params=params,
                           model_state=model_state, PRNGkey=next_key, **self_play_args))

            for thd in as_completed(res):
                px_traj, po_traj = thd.result()
                replay_buffer.add_traj(px_traj)
                replay_buffer.add_traj(po_traj)

        for _ in range(iters_per_train):
            feature, search_prob, true_score = replay_buffer.sample_data(
                batch_size)
            next_params, next_model_state, next_opt_state, val_loss, pol_loss = step(
                feature, true_score, search_prob)
            params = next_params
            model_state = next_model_state
            opt_state = next_opt_state

            train_steps += 1
            writer.add_scalar('value loss', val_loss.item(), train_steps)
            writer.add_scalar('policy loss', pol_loss.item(), train_steps)

        if (index + 1) % ckpt_frequency == 0:
            checkpoint = {'params': params, 'model_state': model_state,
                          'opt_state': opt_state, 'replay_buffer': replay_buffer, 'rand_key': rand_key}

            ckpt_path = os.path.join(
                dir_path, f'checkpoints/{train_steps}.pickle')

            with open(ckpt_path, 'wb') as fp:
                dill.dump(checkpoint, fp)


def main():
    with open(os.path.join(dir_path, 'config.yaml'), 'r') as fp:
        config = yaml.safe_load(fp)
    train(**config, last_ckpt=70)


if __name__ == '__main__':
    main()
