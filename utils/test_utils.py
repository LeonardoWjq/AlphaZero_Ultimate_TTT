from env.ultimate_ttt import UltimateTTT
from env.macros import *
import random
def generate_random_game(num_steps: int, seed: int = 0):
    random.seed(seed)
    game = UltimateTTT(None, None)
    for _ in range(num_steps):
        move = random.choice(game.next_valid_moves)
        game.update_state(move)
        if game.outcome != INCOMPLETE:
            return game.get_state()
    
    return game.get_state()


