from .ai import AI
import chess

class Game:
    def __init__(self, model_name, white_is_ai="True", black_is_ai="True", output_svg="True" ):
        # copy params for later reference
        self.white_is_ai = white_is_ai
        self.black_is_ai = black_is_ai

        self.output_svg = "True"

        # initialize game, AI only if needed
        if white_is_ai or black_is_ai:
            self.ai = AI(model_name)
        self.board = chess.Board()

    def play(self):
        pass