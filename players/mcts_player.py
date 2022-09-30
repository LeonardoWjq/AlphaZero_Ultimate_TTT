from mcts.core import MCTS

from players.player import Player
from players.random_player import RandomPlayer


class MCTSPlayer(Player):
    def __init__(self, roll_out_player = None, num_simulation=500, explore_factor=1.4, verbose=False) -> None:
        super().__init__()
        self.player = RandomPlayer() if roll_out_player is None else roll_out_player
        self.mcts_agent = None
        self.num_sim = num_simulation
        self.C = explore_factor
        self.verbose = verbose

    def move(self, state: dict):

        if self.mcts_agent is None:
            self.mcts_agent = MCTS(state, self.player, self.C)
        else:
            self.mcts_agent.truncate(state)

        x_win_rate, o_win_rate, tie_rate = self.mcts_agent.run_simulation(
            self.num_sim)
        if self.verbose:
            print(
                f'MCTS says the probability of X winning is {x_win_rate:.2%}')
            print(
                f'MCTS says the probability of O winning is {o_win_rate:.2%}')
            print(f'MCTS says the probability of tying is {tie_rate:.2%}')

        return self.mcts_agent.move_and_truncate()

    def reset(self):
        self.mcts_agent = None
