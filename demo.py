from environment import UltimateTTT
from player import HumanPlayer, AlphaZeroPlayer, MCTSPlayer

def main():
    p1 = HumanPlayer()
    p2 = AlphaZeroPlayer()
    game = UltimateTTT(p1, p2)
    game.play(display=True)

if __name__ == '__main__':
    main()