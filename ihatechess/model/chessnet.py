import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Dense,
    Flatten,
    Add,
    ReLU,
    LeakyReLU,
)
import tensorflow.keras.backend as K


class ChessNet(Model):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.model = self.build_model()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def build_model(self):
        inputs = tf.keras.Input(shape=(8, 8, 18))

        x = Conv2D(
            filters=256,
            kernel_size=3,
            padding="same",
            data_format="channels_first",
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(40),
        )(inputs)
        x = BatchNormalization(axis=1)(x)
        x = ReLU()(x)

        for _ in range(7):
            res = Conv2D(
                filters=256,
                kernel_size=3,
                padding="same",
                data_format="channels_first",
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(40),
            )(x)
            res = BatchNormalization(axis=1)(res)
            res = ReLU()(res)
            res = Conv2D(
                filters=256,
                kernel_size=3,
                padding="same",
                data_format="channels_first",
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(40),
            )(res)
            res = BatchNormalization(axis=1)(res)
            res = Add()([res, x])
            x = ReLU()(res)

        # For policy output
        p = Conv2D(
            filters=2,
            kernel_size=1,
            data_format="channels_first",
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(40),
        )(x)
        p = BatchNormalization(axis=1)(p)
        p = ReLU()(p)
        p = Flatten()(p)
        p = Dense(
            8 * 8 * 64,
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(40),
        )(p)

        # For value output
        v = Conv2D(
            filters=4,
            kernel_size=1,
            data_format="channels_first",
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(40),
        )(x)
        v = BatchNormalization(axis=1)(v)
        v = ReLU()(v)
        v = Flatten()(v)
        v = Dense(
            256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(40)
        )(v)
        v = Dense(
            1, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(40)
        )(v)

        return Model(inputs=inputs, outputs=[p, v])

    def call(self, inputs):
        return self.model(inputs)

    def masked_loss(
        self, true_policy, true_value, pred_policy, pred_value, legal_moves
    ):
        # Compute the value loss using mean squared error
        value_loss = tf.reduce_mean(tf.square(true_value - pred_value))

        # Compute the policy loss with masking for illegal moves
        masked_true_policy = true_policy * legal_moves
        masked_pred_policy = pred_policy * legal_moves

        # Compute the policy loss only for legal moves
        policy_loss = -tf.reduce_sum(
            masked_true_policy * tf.math.log(masked_pred_policy + 1e-10)
        )

        # Compute the total loss as a combination of value loss and policy loss
        total_loss = value_loss + policy_loss

        return total_loss

    def compile(self, optimizer):
        super(ChessNet, self).compile()
        self.optimizer = optimizer

    def train_step(self, data):
        inputs, targets = data
        true_policy, true_value, legal_moves = targets

        with tf.GradientTape() as tape:
            pred_policy, pred_value = self(inputs, training=True)  # Forward pass
            # Compute the loss value
            loss = self.masked_loss(
                true_policy, true_value, pred_policy, pred_value, legal_moves
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the loss tracker
        self.loss_tracker.update_state(loss)  # Add this line to update the loss tracker

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(true_policy, pred_policy)
        self.compiled_metrics.update_state(true_value, pred_value)

        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": self.loss_tracker.result()})
        return results
