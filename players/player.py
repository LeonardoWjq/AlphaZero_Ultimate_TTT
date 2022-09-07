# base class to inherit
class Player:
    def __init__(self) -> None:
        pass

    def move(self, state: dict):
        raise NotImplementedError
