from deltachess.ai import NewAIWITHOUTTANH

import chess
import numpy as np
import os.path

class AdversialTraining():
    norm_bool = {True: 1, False: -1}

    DEFAULT_STATS = np.array([0,0,0,0,0,0,0,0])
    DEFAULT_VERBOSITY = 0

    def __init__(self, name1, name2):
        if not (isinstance(name1, str) and isinstance(name2, str)):
            raise Exception("Names need to be strings!")

        self.name_white = name1
        self.name_black = name2

        self.ai_white = NewAIWITHOUTTANH(checkpoint_name=name1)
        self.ai_black = NewAIWITHOUTTANH(checkpoint_name=name2)

        self.stats_name = name1 + "_vs_" + name2 + "_stats"

        self.stats = self.load_stats(self.stats_name)

        self.verbosity = AdversialTraining.DEFAULT_VERBOSITY

    def load_stats(self, name):
        if os.path.exists(name + ".npy"):
            return np.load(name + ".npy")
        else:
            return np.array(AdversialTraining.DEFAULT_STATS) # [bool(which network), played games, SwitchNN1/NN2,0,0,0,0,0]

    def save_stats(self, name, stats):
        np.save(name + ".npy", stats)


    def start_training(self, num_games):

        for i in range(0, num_games):
            self.play_training_game()
            if self.last_victory_status == 1 and self.board.turn != self.stats[0]:
                self.stats[0] = not self.stats[0]
                self.stats[2] += 1

        self.stats[1] = self.stats[1] + num_games
        self.save_stats(self.stats_name, self.stats)

        self.ai_white.save_model()
        self.ai_black.save_model()

        print("\n\nPlayed", self.stats[1], "games and turns taken:", self.stats[2])

    def play_training_game(self):
        """performs the game play"""
        self.board = chess.Board()
        self.game_is_on = True

        while not self.board.is_game_over():
            if self.stats[0] == 0:
                self.ai_black.update_with_current_board(board=self.board)
            elif self.stats[0] == 1:
                self.ai_white.update_with_current_board(board=self.board)

            if self.verbosity > 0:
                print(self.board, "\n")
            if self.board.turn:
                next_move = self.ai_white.get_best_move(self.board)
            else:
                next_move = self.ai_black.get_best_move(self.board)
            self.board.push(next_move)

        self.game_is_on = False
        self.last_victory_status = self.board.is_checkmate()
        self.last_training_value = self.last_victory_status

        """
        weiß wird trainiert:
        - schwarz gewinnt
        - 1 als endwert
        
        - draw
        - 0 als endwert
        
        - nn endwert soll -1 sein -> turn =
        """



        if self.last_victory_status == 0:
            self.last_training_value = self.norm_bool[self.stats[0]] * -0.5

        if self.stats[0] == 0:
            self.ai_black.update_with_current_board(self.board, victory_status=self.last_training_value)
            self.ai_black.end_training()
        elif self.stats[0] == 1:
            self.ai_white.update_with_current_board(self.board, victory_status=self.last_training_value)
            self.ai_white.end_training()

        self.report()

    def report(self):
        """Performs status update

        Implement possibility for good monitoring
            --> Cost Function / training progress
            --> num games / remaining
            --> current game play

        Should be in a different thread!
        """

        if not self.game_is_on:
            # print(self.board)
            if self.last_victory_status:
                print("GameOver, Checkmate, Winner:", not self.board.turn)
            else:
                print("GameOver, draw, stats[0]:", bool(self.stats[0]), "last_training_val:", self.last_training_value)
            print("rounds:", self.board.fullmove_number, "board:", self.board.fen())


"""
    Test whether saving the parameters realy works!    
"""


if __name__ == "__main__":
    training = AdversialTraining(name1="white4_corrected_assignment", name2="black4_corrected_assignment")
    training.start_training(1)

