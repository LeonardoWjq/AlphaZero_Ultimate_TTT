import numpy as np
class Policy:
    def get_probs(self, state:dict):
        pass


class RandomPolicy(Policy):
    def get_probs(self, state: dict):
        length = len(state['valid_move'])
        probs = np.ones(length)/length
        return probs
