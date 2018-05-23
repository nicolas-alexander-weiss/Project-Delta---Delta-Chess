class DeltaTF:
    """Base Class for all TF Models.
    It shall ensure the simplicity of the Game Class and increase its modularity, so that different TF models can be
    implemented, yet without requiring updates of the Game class, such as the way the feature vector is derived.

    Training can be implemented several ways: E.g. after one complete game, thus the update function will accumulate the
    training data which will then be used when the end function is called, or incrementally after every move, thus
    training happens with each call of the update function.

    Variations possible."""

    def __init__(self):
        """Loads saved model from disk or instantiates new model. Parameters might be stored in specific file or
        with TF mechanism."""
        raise NotImplementedError()

    def get_best_move(self, board):
        """returns the best move - according to the model - based on the given board for the given player."""
        raise NotImplementedError()

    def update_with_current_board(self, board):
        """retrieves the current board and extracts the feature vector. May already perform an increment in training or
        accumulate training data for later use."""
        raise NotImplementedError()

    def end_training(self):
        """Performs training if not done earlier."""
        raise NotImplementedError()

    def save_model(self):
        """Saves model to disk. Maybe it should destroy the model"""
        raise NotImplementedError()


class DeltaTF1(DeltaTF):
    def __init__(self):
        print("NOT YET IMPLEMTED")