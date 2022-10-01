from collections import deque
from typing import List
from alphazero.record import Record
import random
import jax
import jax.numpy as jnp


class ReplayBuffer:
    def __init__(self, seed: int, capacity=10000) -> None:
        assert capacity > 0, 'capacity can not be smaller or equal to 0'
        self.buffer = deque([], maxlen=capacity)
        random.seed(seed)

    def add_traj(self, trajectory: List[Record]):
        self.buffer.extend(trajectory)

    def sample_data(self, size):
        buffer_list: List[Record] = random.sample(
            self.buffer, min(size, len(self.buffer)))

        feature_list = []
        prob_list = []
        score_list = []

        for record in buffer_list:
            feature_list.append(record.feature)
            prob_list.append(record.search_prob)
            score_list.append(record.true_score)
        
        feature_array = jnp.concatenate(feature_list)
        prob_array = jnp.stack(prob_list)
        score_array = jnp.vstack(score_list)

        return feature_array, prob_array, score_array


# def main():
#     buffer = ReplayBuffer(0)
#     buffer.add_traj([Record(None, None)])
#     data = buffer.sample_data(1)
#     print(type(data))


# if __name__ == '__main__':
#     main()
