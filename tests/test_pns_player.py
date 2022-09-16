from env.ultimate_ttt import UltimateTTT
from players.pns_player import PNSPlayer
from players.human_player import HumanPlayer
from utils.test_utils import generate_random_game


def test_pns_player(rollout_num=55, seed=0):
    random_state = generate_random_game(rollout_num, seed)
    px = HumanPlayer()
    po = PNSPlayer(verbose=True)
    game = UltimateTTT(px, po, random_state)
    game.play(display='all')


if __name__ == '__main__':
    test_pns_player(45, 0)