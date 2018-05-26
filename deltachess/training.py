from .ai import TestModel
import chess


class SelfPlayTraining:
    def __init__(self, num_games=1, verbosity=1):
        """Initiates the AI model as well as training parameters"""
        self.ai = TestModel()
        self.num_games = num_games
        self.num_remaining_games = num_games
        self.verbosity = verbosity
        self.board = None
        self.game_is_on = False

        self.last_victory_status = None
        self.stats = [0,0,0,0] # [total_games, num_checkmates, num_draws, num_games_won_by_white]

    def start_training(self):
        """calls training method for i in range(0, num_games)"""
        while self.num_remaining_games > 0:
            self.play_training_game()
            self.num_remaining_games = self.num_remaining_games - 1

    def play_training_game(self):
        """performs the game play"""
        self.board = chess.Board()
        self.game_is_on = True

        while not self.board.is_game_over():
            if self.verbosity > 0:
                self.update_progress()
            next_move = self.ai.get_best_move(self.board)
            self.board.push(next_move)

        self.game_is_on = False
        self.last_victory_status = self.board.is_checkmate()
        self.ai.update_with_current_board(self.board, victory_status=self.last_victory_status)
        self.update_progress()

    def update_progress(self):
        """Performs update of status in dependence on level of verbosity.
        Maybe: Website which presents training status, board etc.

        Implement possibility for good monitoring
            --> Cost Function / training progress
            --> num games / remaining
            --> current game play

        Should be in a different thread!
        """
        if not self.game_is_on:
            print(self.stats)
            if self.last_victory_status:
                print("GameOver, Checkmate, Winner:", not self.board.turn)
            else:
                print("GameOver, draw")
            print("rounds:", self.board.fullmove_number, "board:", self.board.fen())



