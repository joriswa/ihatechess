from multiprocessing import Pool, Manager
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from ihatechess.chess.board import Board

from ihatechess.mcts.mcts import MCTS


class Trainer:
    def __init__(
        self,
        network,
        num_games,
        num_simulations,
        batch_size,
        num_epochs,
        learning_rate=0.01,
    ):
        self.network = network
        self.num_games = num_games
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = Adam(learning_rate)

    def train(self):
        with Manager() as manager:
            # Create a list in shared memory
            results = manager.list()

            # Use a Pool of workers to play games in parallel
            with Pool() as pool:
                args = [
                    (self.network, self.num_simulations, results)
                    for _ in range(self.num_games)
                ]
                pool.starmap_async(play_game, args)

            # Collect the games and add them to the dataset
            dataset = list(results)

            # Train the network using the collected games
            for epoch in range(self.num_epochs):
                # Shuffle dataset
                np.random.shuffle(dataset)

                for i in range(0, len(dataset), self.batch_size):
                    batch = dataset[i : i + self.batch_size]

                    # Extract the boards and the corresponding targets from the batch
                    boards, policy_targets, value_targets = zip(*batch)

                    # Convert boards and targets into TensorFlow tensors
                    boards = tf.convert_to_tensor(boards, dtype=tf.float32)
                    policy_targets = tf.convert_to_tensor(
                        policy_targets, dtype=tf.float32
                    )
                    value_targets = tf.convert_to_tensor(
                        value_targets, dtype=tf.float32
                    )

                    self.train_step(boards, policy_targets, value_targets)

    def train_step(self, boards, policy_targets, value_targets):
        with tf.GradientTape() as tape:
            # Forward pass
            policy_preds, value_preds = self.network(boards)

            # Compute loss
            policy_loss = tf.keras.losses.categorical_crossentropy(
                policy_targets, policy_preds
            )
            value_loss = tf.keras.losses.mean_squared_error(value_targets, value_preds)
            loss = policy_loss + value_loss

        # Backward pass and optimization
        grads = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))


def play_game(network, num_simulations, results):
    mcts = MCTS(network, num_simulations)
    game_result = mcts.generate_game_data(Board())
    results.append(game_result)
