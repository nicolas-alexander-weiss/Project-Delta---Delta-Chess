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
    of each type and the board itself. Of course, all features are scaled to a range of [-1;1].

    Updating parameters using temporal difference learning, incrementing with every step. Expected to be very slow.
    """

    # helper dicts:
    sym_to_int = {"P": chess.PAWN, "N": chess.KNIGHT, "B": chess.BISHOP, "R": chess.ROOK, "Q": chess.QUEEN,
                  "K": chess.KING,
                  "p": - chess.PAWN, "n": - chess.KNIGHT, "b": - chess.BISHOP, "r": - chess.ROOK,
                  "q": - chess.QUEEN,"k": - chess.KING,
                  "None": 0}
    norm_bool = {True: 1, False: -1}

    # Turn indicator (1), castling rights(4), en passant possible (1), Num of each piece (also empty fields count) (13)
    # , Board (64)
    num_features = 83

    # learning parameters
    alpha = 0.1
    lambda_discount = 0.3

    @staticmethod
    def get_feature_vector(board):

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

        for i in range(0, 64, 1):
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

    def __init__(self, name="undefined"):

        # create timestamp if undefined -> used as checkpoint name for TF
        if name == "undefined":
            self.checkpoint_name = str(time.strftime("%c"))
        else:
            self.checkpoint_name = name


        #

        # build graph

        self.tf_input_vec = tf.placeholder(tf.float64, shape=[None, 83], name="input_vec")

        # split into board / other features -> input layer
        self.tf_global_features_vec, self.tf_board_feature_vec = tf.split(self.tf_input_vec, [19, 64], 1)

        # First Hidden Layer
        self.tf_layer_1_1_w = tf.Variable(initial_value=tf.random_normal(dtype=tf.float64 ,shape=(19, 19), stddev=0.1))
        self.tf_layer_1_1_b = tf.Variable(initial_value=tf.random_normal(dtype=tf.float64 ,shape=(19,), stddev=0.1))
        self.tf_layer_1_2_w = tf.Variable(initial_value=tf.random_normal(dtype=tf.float64 ,shape=(64, 64), stddev=0.1))
        self.tf_layer_1_2_b = tf.Variable(initial_value=tf.random_normal(dtype=tf.float64 ,shape=(64,), stddev=0.1))

        # get output of first hidden layer, stacking in order to pass them together through last layer
        self.tf_weighted_sum_layer_1_1 = self.tf_layer_1_1_b + self.tf_global_features_vec @ tf.transpose(self.tf_layer_1_1_w)
        self.tf_weighted_sum_layer_1_2 = self.tf_layer_1_2_b + self.tf_board_feature_vec @ tf.transpose(self.tf_layer_1_2_w)

        self.tf_output_layer_1_1 = tf.nn.relu(self.tf_weighted_sum_layer_1_1)
        self.tf_output_layer_1_2 = tf.nn.relu(self.tf_weighted_sum_layer_1_2)

        # combine, for next layer
        self.tf_input_layer_2 = tf.concat([self.tf_output_layer_1_1,
                                           self.tf_output_layer_1_2],
                                          axis=1)

        # Second HL
        self.tf_layer_2_w = tf.Variable(initial_value=tf.random_normal(dtype=tf.float64, shape=(83, 83), stddev=0.1))
        self.tf_layer_2_b = tf.Variable(initial_value=tf.random_normal(dtype=tf.float64, shape=(83,), stddev=0.1))

        # pass through
        self.tf_weighted_sum_layer_2 = self.tf_layer_2_b + self.tf_input_layer_2 @ tf.transpose(self.tf_layer_2_w)
        self.tf_input_layer_3 = tf.nn.relu(self.tf_weighted_sum_layer_2)

        # output layer
        self.tf_layer_3_w = tf.Variable(initial_value=tf.random_normal(dtype=tf.float64, shape=(1, 83), stddev=0.1))
        self.tf_layer_3_b = tf.Variable(initial_value=tf.random_normal(dtype=tf.float64, shape=(1,), stddev=0.1))

        # get output
        self.tf_weighted_sum_layer_3 = self.tf_layer_3_b + self.tf_input_layer_3 @ tf.transpose(self.tf_layer_3_w)
        self.tf_output = tf.nn.tanh(self.tf_weighted_sum_layer_3)

        #
        # Learning:
        #

        # learning parameters

        self.tf_alpha = tf.placeholder(tf.float32, shape=(1,))
        self.tf_lambda = tf.placeholder(tf.float32, shape=(1,))

        # saving feature vecs

        self.tf_saved_feature_vec_stack = tf.Variable(initial_value=tf.zeros((1000, 83), dtype=tf.float64))
        self.tf_num_saved_feature_vec = tf.Variable(initial_value=0)
        self.num_saved_feature_vecs = 0
        # self.tf_get_saved_feature_vecs = self.tf_saved_feature_vec_stack[self.tf_num_saved_feature_vec,:]

        self.tf_op_add_feature_vec = tf.assign(self.tf_saved_feature_vec_stack,
                                               tf.add(self.tf_saved_feature_vec_stack,
                                                      tf.matmul(tf.reshape(tf.one_hot(self.tf_num_saved_feature_vec, 1000, dtype=tf.float64),
                                                                           shape=(1000,1)),
                                                      self.tf_input_vec)))

        self.tf_increment_tf_num_saved_feature_vec = tf.assign(self.tf_num_saved_feature_vec,
                                                               tf.add(self.tf_num_saved_feature_vec, 1))

        # Create gradient and update operations

        self.tf_grad_1_1_w, self.tf_grad_1_1_b\
            , self.tf_grad_1_2_w, self.tf_grad_1_2_b\
            , self.tf_grad_2_w, self.tf_grad_2_b\
            , self.tf_grad_3_w, self.tf_grad_3_b\
            = tf.gradients(self.tf_output, [self.tf_layer_1_1_w, self.tf_layer_1_1_b,
                                            self.tf_layer_1_2_w, self.tf_layer_1_2_b,
                                            self.tf_layer_2_w, self.tf_layer_2_b,
                                            self.tf_layer_3_w, self.tf_layer_3_b])

        # self.tf_num_input_vecs = tf.shape(self.tf_input_vec)[1]

        self.tf_all_gradients = tf.concat(
            [
                tf.reshape(self.tf_grad_1_1_w, shape=(-1, 19*19)),
                tf.reshape(self.tf_grad_1_1_b, shape=(-1, 19)),
                tf.reshape(self.tf_grad_1_2_w, shape=(-1, 64*64)),
                tf.reshape(self.tf_grad_1_2_b, shape=(-1, 64)),
                tf.reshape(self.tf_grad_2_w, shape=(-1, 83*83)),
                tf.reshape(self.tf_grad_2_b, shape=(-1, 83)),
                tf.reshape(self.tf_grad_3_w, shape=(-1, 83)),
                tf.reshape(self.tf_grad_3_b, shape=(-1, 1))
            ],
            axis=1
        )

        # compute gradients on all input

        #
        #

        # init operators
        self.tf_var_init = tf.global_variables_initializer()

        # create saver and session
        self.tf_saver = tf.train.Saver()
        self.tf_sess = tf.Session()

        # load variables if checkpoint exists, otherwise initialize variables
        if tf.train.checkpoint_exists(self.checkpoint_name):
            self.saver.restore(sess=self.sess)
        else:
            # writer = tf.summary.FileWriter(".", self.tf_sess.graph)
            self.tf_sess.run((self.tf_var_init,))
            # writer.close()
            """
            print(self.tf_sess.run((self.tf_input_layer_2,
                                    self.tf_input_layer_3,
                                    self.tf_output),
                                   feed_dict={self.tf_input_vec: [self.get_feature_vector(chess.Board())]}))
            """

    def update_with_current_board(self, board, victory_status=-1):

        self.tf_sess.run(self.tf_op_add_feature_vec,
                         feed_dict={self.tf_input_vec: [SEOTDNN.get_feature_vector(board)]})
        self.tf_sess.run(self.tf_increment_tf_num_saved_feature_vec)

        self.num_saved_feature_vecs += 1

        if self.num_saved_feature_vecs >= 2:
            self.update_weights_with_saved_feature_vecs(victory_status, board.turn)

    def update_weights_with_saved_feature_vecs(self, victory_status, at_turn):

        newest_feature_vec = self.tf_sess.run(self.tf_saved_feature_vec_stack[self.num_saved_feature_vecs - 1, :])
        old_feature_vecs = self.tf_sess.run(self.tf_saved_feature_vec_stack[0:self.num_saved_feature_vecs - 1, :])

        all_gradients = np.array([
            self.tf_sess.run(self.tf_all_gradients, feed_dict={self.tf_input_vec: [vec]})
             for vec in old_feature_vecs])

        lambda_powers = np.reshape(np.power(self.lambda_discount, range(self.num_saved_feature_vecs - 2, -1, -1)),
                                   newshape=(self.num_saved_feature_vecs-1,1))
        # print("lamdbas", lambda_powers)
        all_gradients = np.reshape(all_gradients, newshape=(self.num_saved_feature_vecs - 1, 11596))
        # print("all_gradients", all_gradients)

        multiply = self.tf_sess.run(tf.multiply(lambda_powers,all_gradients))
        # print("multiply", multiply)

        reduce_sum = self.tf_sess.run(tf.reduce_sum(multiply, axis=0))
        # print("reducesum", reduce_sum)

        pred1 = victory_status
        if victory_status != -1:
            pred1 = victory_status * SEOTDNN.norm_bool[at_turn]
            print("pred1", pred1)
        else:
            self.tf_sess.run(self.tf_output, feed_dict={self.tf_input_vec: [newest_feature_vec]})

        pred2 = self.tf_sess.run(self.tf_output, feed_dict={self.tf_input_vec: [old_feature_vecs[-1]]})
        pred_diff = pred1 - pred2
        update_val = self.alpha * pred_diff * reduce_sum

        #print(update_val)
        self.update_weights(update_val)

    def update_weights(self, update_val):
        print(update_val.shape)
        w1_1,b1_1,w1_2,b1_2,w2,b2,w3,b3 = tf.split(update_val, [19*19,19,64*64,64,83*83,83, 83,1], axis=1)
        u_w1_1 = tf.assign(self.tf_layer_1_1_w, tf.add(self.tf_layer_1_1_w, tf.reshape(w1_1,(19,19))))
        u_b1_1 = tf.assign(self.tf_layer_1_1_b, tf.add(self.tf_layer_1_1_b, tf.reshape(b1_1, (19,))))
        u_w1_2 = tf.assign(self.tf_layer_1_2_w, tf.add(self.tf_layer_1_2_w, tf.reshape(w1_2, (64,64))))
        u_b1_2 = tf.assign(self.tf_layer_1_2_b, tf.add(self.tf_layer_1_2_b, tf.reshape(b1_2, (64,))))
        u_w2 = tf.assign(self.tf_layer_2_w, tf.add(self.tf_layer_2_w, tf.reshape(w2, (83,83))))
        u_b2 = tf.assign(self.tf_layer_2_b, tf.add(self.tf_layer_2_b, tf.reshape(b2, (83,))))
        u_w3 = tf.assign(self.tf_layer_3_w, tf.add(self.tf_layer_3_w, tf.reshape(w3, (1,83))))
        u_b3 = tf.assign(self.tf_layer_3_b, tf.add(self.tf_layer_3_b, tf.reshape(b3, (1,))))

        self.tf_sess.run((u_w1_1,u_b1_1,u_w1_2,u_b1_2,u_w2,u_b2,u_w3,u_b3))

    def end_training(self, board):
        pass

    def get_best_move(self, board):
        for move in board.legal_moves:
            return move



    def pass_through(self, feature_vec):
        return self.tf_sess.run(self.tf_output, feed_dict={self.tf_input_vec: feature_vec})


    def save_model(self):
        self.tf_saver.save(sess=self.tf_sess, save_path=self.checkpoint_name)

    def close_model(self):
        self.tf_saver.save(sess=self.tf_sess, save_path=self.checkpoint_name)
        print("Checkpoint name:", self.checkpoint_name)
        self.tf_sess.close()



if __name__ == "__main__":
    board = chess.Board()
    ai = SEOTDNN()

    # print(ai.tf_sess.run((ai.tf_output, ai.tf_grad_3_b), feed_dict={ai.tf_input_vec:[SEOTDNN.get_feature_vector(board),SEOTDNN.get_feature_vector(board)]}))
    # print(ai.tf_sess.run((ai.tf_output, ai.tf_grad_3_b), feed_dict={
    #     ai.tf_input_vec: [SEOTDNN.get_feature_vector(board)]}))

    ai.update_with_current_board(board)

    board.push_san("Na3")

    ai.update_with_current_board(board)

    #board.push_san("Na6")

    #ai.update_with_current_board(board)
    #print(board.piece_at(0).symbol())
    #print(SEOTDNN.get_feature_vector(board))

    """
    print(self.tf_sess.run((self.tf_input_layer_2,
                                    self.tf_input_layer_3,
                                    self.tf_output), feed_dict={self.tf_input_vec: [self.get_feature_vector(chess.Board())]}))

    """


