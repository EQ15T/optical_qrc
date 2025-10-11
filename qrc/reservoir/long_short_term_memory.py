import numpy as np
import tensorflow as tf
from typing import Optional

from .abstract_reservoir import AbstractReservoir

tf.get_logger().setLevel("ERROR")


class LongShortTermMemory(AbstractReservoir):
    def __init__(
        self,
        input_dim,
        out_dim,
        hidden_dim,
        epochs,
        learning_rate,
        dropout_rate,
        seed=None,
        show_progress=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout_rate
        self.lr = learning_rate
        self.epochs = epochs
        self.show_progress = show_progress

        if seed is not None:
            tf.random.set_seed(seed)

        # Sequential model with single LSTM layer + Dense readout
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(batch_shape=(1, None, input_dim)),
                tf.keras.layers.LSTM(
                    hidden_dim,
                    return_sequences=True,
                    dropout=self.dropout,
                    recurrent_dropout=self.dropout,
                ),
                tf.keras.layers.Dense(out_dim),  # Linear readout
            ]
        )

    @property
    def input_dimension(self):
        return self.input_dim

    @property
    def output_dimension(self):
        return self.hidden_dim

    @property
    def num_parameters(self):
        breakdown = {}
        for layer in self.model.layers:
            breakdown[layer.name] = layer.count_params()

        # Analytical count for verification
        # LSTM: 4*(hidden_dim*(hidden_dim + input_dim + 1))
        lstm_params = 4 * self.hidden_dim * (self.hidden_dim + self.input_dim + 1)
        # Dense: (hidden_dim + 1) * out_dim
        dense_params = (self.hidden_dim + 1) * self.out_dim
        total_params = lstm_params + dense_params

        breakdown["total (Analytical)"] = total_params

        return breakdown

    def step(self, s: np.ndarray) -> np.ndarray:
        """Compatibility method, not used in LSTM context."""
        return None  # Not used in LSTM context

    def reset(self, seed=None):
        """
        Rebuild the full LSTM model from scratch.
        """
        if seed is not None:
            tf.random.set_seed(seed)

        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(batch_shape=(1, None, self.input_dim)),
                tf.keras.layers.LSTM(
                    self.hidden_dim,
                    return_sequences=True,
                    dropout=self.dropout,
                    recurrent_dropout=self.dropout,
                ),
                tf.keras.layers.Dense(self.out_dim),
            ]
        )

        self._trained = False

    def train(self, x_train, y_train, washout=0):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss="mse",  # Standard MSE loss
        )

        if x_train.ndim == 1:
            x_train = x_train[:, None]
        if y_train.ndim == 1:
            y_train = y_train[:, None]

        x_train = x_train[None, :, :]  # batch dimension
        y_train = y_train[None, :, :]

        self.model.fit(
            x_train,
            y_train,
            epochs=self.epochs,
            verbose=1 if self.show_progress else 0,
            batch_size=1,
        )
        self._trained = True

    def predict(self, x_seq):
        if x_seq.ndim == 1:
            x_seq = x_seq[:, None]
        x_seq = x_seq[None, :, :]  # batch dimension

        y_pred = self.model.predict(
            x_seq,
            verbose=0,
            batch_size=1,
        )
        return y_pred[0]  # remove batch dimension

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "out_dim": self.out_dim,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "learning_rate": self.lr,
            "dropout_rate": self.dropout,
            "show_progress": self.show_progress,
        }
