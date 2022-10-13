from collections import deque

from env.ultimate_ttt import UltimateTTT
from jax import jit, random
from utils.alphazero_utils import get_move_probs
from utils.test_utils import generate_random_game

from alphazero.model import create_model, init_model
from alphazero.node import Node
from alphazero.record import Record


class AlphaZero:
    def __init__(self, model_params, model_state, PRNGkey, sim_num: int, explore_factor: float, temperature: float, alpha, epsilon) -> None:
        model = create_model(training=False)

        def forward(feature):
            (val, logits), _ = model.apply(model_params, model_state, feature)
            return val, logits
        self.forward_func = jit(forward)
        self.sim_num = sim_num
        self.history = deque([], maxlen=8)
        self.traj_record = []
        self.C = explore_factor
        self.tau = temperature
        self.key = PRNGkey
        self.alpha = alpha
        self.epsilon = epsilon

    def get_move(self, state: dict):
        self.history.appendleft(state)
        self.key, node_key, move_key = random.split(self.key, num=3)

        node = Node(state, self.history, self.forward_func,
                    self.C, True, node_key, self.alpha, self.epsilon)
        for _ in range(self.sim_num):
            node.simulate()

        moves, visit_counts = node.get_dist()
        move_probs = get_move_probs(moves, visit_counts, self.tau)

        # select move
        move = random.choice(move_key, 81, (), p=move_probs)

        feature = node.get_feature()
        record = Record(feature=feature, search_prob=move_probs)
        self.traj_record.append(record)

        # compute the next game state after making the move
        game = UltimateTTT(None, None, state)
        game.update_state(move)
        next_state = game.get_state()
        self.history.appendleft(next_state)

        return move

    def get_traj_record(self):
        return self.traj_record


def main():
    model = create_model(True)
    params, state = init_model(model)
    key = random.PRNGKey(0)
    alpha = AlphaZero(params, state, key, 50, 3.5, 1.0, 0.5, 0.25)
    game = generate_random_game(30)
    move = alpha.get_move(game)
    print(move)


if __name__ == '__main__':
    main()
