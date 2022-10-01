from dataclasses import dataclass
import jax.numpy as jnp
@dataclass
class Record:
    feature: jnp.ndarray
    search_prob: jnp.ndarray
    true_score: float = None

    def set_score(self, score):
        self.true_score = score