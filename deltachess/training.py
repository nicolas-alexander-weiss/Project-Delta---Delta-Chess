from ai import NewAI
import chess


class SelfPlayTraining:
    def __init__(self, save_name="undefined", num_games=1, verbosity=1):
        """Initiates the AI model as well as training parameters"""
        self.ai = NewAI(save_name, learning_rate = 0.1)
        self.num_games = num_games
        self.num_remaining_games = num_games
        self.verbosity = verbosity
        self.board = None
        self.game_is_on = False
        self.last_victory_status = None

    def start_training(self):
        """calls training method for i in range(0, num_games)"""
        while self.num_remaining_games > 0:
            self.play_training_game()
            self.num_remaining_games = self.num_remaining_games - 1
            # print("w3", spt.ai.tf_sess.run(spt.ai.tf_w3))

        self.ai.close_model()


    def play_training_game(self):
        """performs the game play"""
        self.board = chess.Board()
        self.game_is_on = True

        while not self.board.is_game_over():
            self.ai.update_with_current_board(board=self.board)
            if self.verbosity > 0:
                print(self.board, "\n")
            next_move = self.ai.get_best_move(self.board)
            self.board.push(next_move)

        self.game_is_on = False
        self.last_victory_status = self.board.is_checkmate()
        self.ai.update_with_current_board(self.board, victory_status=self.last_victory_status)
        self.ai.end_training()

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
                print("GameOver, draw")
            print("rounds:", self.board.fullmove_number, "board:", self.board.fen())


if __name__ == "__main__":
    spt = SelfPlayTraining(save_name="second_try", num_games=1, verbosity=0)
    spt.start_training()


