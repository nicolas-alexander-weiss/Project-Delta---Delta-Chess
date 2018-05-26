from abc import ABCMeta, abstractmethod
import os
import time

import tensorflow as tf
import numpy as np

import chess


class AI(metaclass=ABCMeta):
    """Base Class for all AI Models.
    It shall ensure the simplicity of the Game Class and increase its modularity, so that different AI models can be
    implemented, yet without requiring updates of the Game class, such as the way the feature vector is derived.

    Training can be implemented several ways: E.g. after one complete game, thus the update function will accumulate the
    training data which will then be used when the end function is called, or incrementally after every move, thus
    training happens with each call of the update function.

    Variations possible."""

    @abstractmethod
    def __init__(self):
        """Loads saved model from disk or instantiates new model. Parameters might be stored in specific file or
        with TF mechanism."""
        pass
    @abstractmethod
    def get_best_move(self, board):
        """returns the best move - according to the model - based on the given board for the given player."""
        pass
    @abstractmethod
    def update_with_current_board(self, board, victory_status=-1):
        """retrieves the current board and extracts the feature vector. May already perform an increment in training or
        accumulate training data for later use."""
        pass
    @abstractmethod
    def end_training(self, board):
        """Performs training if not done earlier."""
        pass
    @abstractmethod
    def save_model(self):
        """Saves model to disk. """
        pass
    @abstractmethod
    def close_model(self):
        """Maybe it should destroy the object? If TF -> close session!"""
        pass


class TestModel(AI):
    def __init__(self):
        print("TEST MODEL - NOT FOR OFFICIAL USE!")

    def get_best_move(self, board):
        for move in board.legal_moves:
            return move

    def update_with_current_board(self, board, victory_status=-1):
        pass
    def end_training(self, board):
        pass
    def save_model(self):
        pass
    def close_model(self):
        pass


class SEOTDNN(AI):
    """StaticEvaluationOnlyWithTDAndNN
    Only uses features which are quite easily computable (turn indicator, castling rights, en passant, num pieces
    of each type and the board itself. Of course, all features are scaled to a range of [-1;1]."""

    def update_with_current_board(self, board, victory_status=-1):
        pass

    def end_training(self, board):
        pass

    def save_model(self):
        pass

    def get_best_move(self, board):
        pass

    # helper dict:
    sym_to_int = {"P": chess.PAWN, "N": chess.KNIGHT, "B": chess.BISHOP, "R": chess.ROOK, "Q": chess.QUEEN,
                  "K": chess.KING,
                  "p": - chess.PAWN, "n": - chess.KNIGHT, "b": - chess.BISHOP, "r": - chess.ROOK,
                  "q": - chess.QUEEN,"k": - chess.KING,
                  "None": 0}
    norm_bool = {True: 1, False: -1}

    # Turn indicator (1), castling rights(4), en passant possible (1), Num of each piece (also empty fields count) (13)
    # , Board (64)
    num_features = 83

    def __init__(self, name="undefined"):

        # create timestamp if undefined -> used as checkpoint name for TF
        if name == "undefined":
            self.checkpoint_name = str(time.strftime("%c"))
        else:
            self.checkpoint_name = name

        # build graph

        # init operators
        self.var_init = tf.global_variables_initializer()

        # create saver and session
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        # load variables if checkpoint exists, otherwise initialize variables
        if tf.train.checkpoint_exists(self.checkpoint_name):
            self.saver.restore(sess=self.sess)
        else:
            self.sess.run((self.var_init,))

    def close_model(self):
        self.saver.save(sess=self.sess, save_path=self.checkpoint_name)
        print("Checkpoint name:", self.checkpoint_name)
        self.sess.close()

    @staticmethod
    def get_feature_vector(board):

        print(board)
        # sanity check
        if not isinstance(board, chess.Board):
            raise ValueError("Parameter is not of instance", chess.Board)

        vec = np.zeros(SEOTDNN.num_features)

        # [-1,1] normalized
        vec[0] = SEOTDNN.norm_bool[board.turn]
        vec[1] = SEOTDNN.norm_bool[board.has_queenside_castling_rights(True)]
        vec[2] = SEOTDNN.norm_bool[board.has_kingside_castling_rights(True)]
        vec[3] = SEOTDNN.norm_bool[board.has_queenside_castling_rights(False)]
        vec[4] = SEOTDNN.norm_bool[board.has_kingside_castling_rights(False)]
        vec[5] = SEOTDNN.norm_bool[board.has_legal_en_passant()]
        # vec[6:18]: Piece counts
        # vec[19:82]: Board positions

        for i in range(0,64,1):

            piece_val = SEOTDNN.sym_to_int[str(board.piece_at(i))]

            vec[SEOTDNN.get_piece_count_index(piece_val)] += 1

            # already zero mean, just normalize
            vec[19 + i] = piece_val / 6

        # scale piece counts
        vec[6:11] -= 4
        vec[6:11] /= 4
        vec[12] -= 32
        vec[12] /= 32
        vec[13:18] -= 4
        vec[13:18] /= 4

        return vec

    @staticmethod
    def get_piece_count_index(piece_val):
        if piece_val < -6 or piece_val > 6:
            raise ValueError("Piece is not in range [-6,6]")

        return 12 + piece_val


if __name__ == "__main__":
    board = chess.Board()
    #print(board.piece_at(0).symbol())
    print(SEOTDNN.get_feature_vector(board))


