from typing import List, Tuple
import numpy as np
import onnxruntime as rt
from ihatechess.chess.board import Board
from ihatechess.chess.utils import move_to_uci
from ihatechess.model.converter import encode_board, encode_move
import tensorflow as tf

EPS = 1e-8  # small constant to avoid division by zero


class MCTS:
    """
    The MCTS (Monte Carlo Tree Search) class provides an implementation of Monte Carlo Tree Search algorithm.
    This class is used to simulate the chess game and choose the best move.

    Args:
        inference (OnnxInferenceSession): ONNX Runtime Inference session for evaluating the chess game state.
        num_simulations (int): Number of MCTS simulations per move.
        cpuct (float, optional): Exploration-exploitation factor. Defaults to 1.0.
        temp_initial (float, optional): Initial temperature. Defaults to 1.0.
        temp_final (float, optional): Final temperature. Defaults to 0.01.
        temp_threshold (int, optional): Threshold for temperature drop. Defaults to 30.
        dirichlet_alpha (float, optional): Alpha parameter for Dirichlet noise. Defaults to 0.3.
        dirichlet_eps (float, optional): Fraction of Dirichlet noise to mix with original policy. Defaults to 0.25.

    Attributes:
        moves (dict): Dictionary to store the moves made in each game state.
        policy (dict): Dictionary to store the policy for each game state.
        values (dict): Dictionary to store the values of each game state.
        visit_counts (dict): Dictionary to store the visit counts of each move in each game state.
        progressive_widening_threshold (int): Threshold for limiting the number of legal moves considered at each node.
    """

    def __init__(
        self,
        inference: rt.InferenceSession,
        num_simulations: int,
        cpuct: float = 1.0,
        temp_initial: float = 1.0,
        temp_final: float = 0.01,
        temp_threshold: float = 30,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
    ):
        self.inference = inference
        self.input_name = self.inference.get_inputs()[0].name
        self.num_simulations = num_simulations

        self.moves = {}
        self.policy = {}
        self.values = {}
        self.visit_counts = {}

        self.cpuct = cpuct
        self.temp_initial = temp_initial
        self.temp_final = temp_final
        self.temp_threshold = temp_threshold
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.progressive_widening_threshold = 7

    def get_move(self, board: Board) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Uses MCTS to find and return the best move according to the highest visit count.

        Args:
            board (Board): Current state of the chess board.

        Returns:
            move: The best move according to the highest visit count.
        """

        self.moves = {}
        self.values = {}
        self.policy = {}
        self.visit_counts = {}

        legal_moves = board.generate_legal_moves()

        # Perform MCTS simulations
        for _ in range(self.num_simulations):
            self._run_simulation(board.copy(), legal_moves)

        # Get visit counts of legal moves
        visit_counts = [
            self.visit_counts.get((board.to_fen(), move_to_uci(move)), 0)
            for move in legal_moves
        ]

        # Apply temperature to visit counts and use softmax to get move probabilities
        temp = (
            self.temp_initial
            if len(board.previous_boards) > self.temp_threshold
            else self.temp_final
        )

        counts = np.array(visit_counts)
        probs = counts ** (1 / temp)
        probs /= probs.sum()

        # Choose a move according to the adjusted probabilities
        move_idx = np.random.choice(len(legal_moves), p=probs)
        move = legal_moves[move_idx]

        return move

    def _run_simulation(
        self, board: Board, legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    ) -> float:
        """
        Simulates a game to evaluate the value of a state and backpropagates the value to update the game tree.

        Args:
            board (Board): Current state of the chess board.
            legal_moves (List[Tuple[Tuple[int, int], Tuple[int, int]]]): List of legal moves.

        Returns:
            float: Value of the board state.
        """

        path = []

        # Play a game until a new state is encountered and compute the value either from the game state or by querying the network
        value = self._explore(board, legal_moves, path)

        # Backpropagate the value
        self._backpropagate(path, value)

    def _explore(self, board, legal_moves, path):
        """
        Explores the game tree until a leaf node is encountered and then returns the value of the leaf node.

        Args:
            board (Board): Current state of the chess board.
            legal_moves (List[Tuple[Tuple[int, int], Tuple[int, int]]]): List of legal moves.
            path (List): Path followed from the root to the current node.

        Returns:
            float: Value of the leaf node.
        """

        while True:
            state = board.to_fen()

            # If game is over, store the terminal value and break the loop
            if board.is_game_over():
                turn = 1 if board.active_color == "white" else -1
                value = board.evaluate() * turn
                break

            # If the policy for the state is not known, query the network
            if state not in self.policy:
                input_data = np.expand_dims(encode_board(board), axis=0)
                probs, value = self.inference.run(None, {self.input_name: input_data})
                probs = tf.reshape(probs, (8, 8, 64))
                self.policy[state] = {
                    move_to_uci(move): probs[*encode_move(move)] for move in legal_moves
                }

                # Add Dirichlet noise to the root node
                if len(self.visit_counts) == 0:
                    noise = np.random.dirichlet(
                        [self.dirichlet_alpha] * len(legal_moves)
                    )
                    self.policy[state] = {
                        (move_to_uci(move)): (1 - self.dirichlet_eps)
                        * self.policy[state][move_to_uci(move)]
                        + self.dirichlet_eps * noise[i]
                        for i, move in enumerate(legal_moves)
                    }

                break

            # Select a move according to the UCB formula
            move = self._select_move(state, legal_moves)
            board.move_piece(*move)
            path.append((state, move_to_uci(move)))

            # Progressive Widening: Limit the number of legal moves considered at each node
            legal_moves = board.generate_legal_moves()
            if len(legal_moves) > self.progressive_widening_threshold:
                legal_moves = self._order_moves(board, legal_moves)[
                    : self.progressive_widening_threshold
                ]

        return value

    def _backpropagate(self, path, value):
        """
        Backpropagates the value of a leaf node to update the values and visit counts of its ancestors in the game tree.

        Args:
            path (List): Path followed from the root to the leaf node.
            value (float): Value of the leaf node.
        """
        for state, move in reversed(path):
            if (state, move) in self.moves:
                # Update the Q-value for the move
                self.moves[(state, move)] = (
                    self.visit_counts[(state, move)] * self.moves[(state, move)] + value
                ) / (self.visit_counts[(state, move)] + 1)
                # Increment the visit count for the move
                self.visit_counts[(state, move)] += 1
            else:
                self.moves[(state, move)] = value
                self.visit_counts[(state, move)] = 1

    def _select_move(self, state, legal_moves):
        """
        Selects a move according to the Upper Confidence Bound (UCB) formula.

        Args:
            state (str): FEN representation of the current chess board state.
            legal_moves (List[Tuple[Tuple[int, int], Tuple[int, int]]]): List of legal moves.

        Returns:
            best_move (Tuple[Tuple[int, int], Tuple[int, int]]): The move with the highest UCB value.
        """

        max_u, best_move = -np.inf, None

        for move in legal_moves:
            uci_move = move_to_uci(move)
            if (state, uci_move) in self.moves:
                q = self.moves[(state, uci_move)]
                p = self.policy[state].get(uci_move, 0)
                n = sum(
                    self.visit_counts.get((state, move_to_uci(m)), 0)
                    for m in legal_moves
                )

                # UCB formula
                u = q + self.cpuct * p * np.sqrt(n) / (
                    1 + self.visit_counts.get((state, uci_move), 0)
                )
            else:
                u = (
                    self.cpuct
                    * self.policy[state].get(uci_move, 0)
                    * np.sqrt(
                        sum(
                            self.visit_counts.get((state, move_to_uci(m)), 0)
                            for m in legal_moves
                        )
                        + EPS
                    )
                )

            if u > max_u:
                max_u = u
                best_move = move

        return best_move

    def _order_moves(
        self, board: Board, legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Orders the legal moves based on the value of the capturing move.

        Args:
            board (Board): Current state of the chess board.
            legal_moves (List[Tuple[Tuple[int, int], Tuple[int, int]]]): List of legal moves.

        Returns:
            sorted_moves (List[Tuple[Tuple[int, int], Tuple[int, int]]]): The sorted list of legal moves.
        """
        sorted_moves = sorted(
            legal_moves,
            key=lambda move: self._get_move_capturing_value(board, move),
            reverse=True,
        )

        return sorted_moves

    def _get_move_capturing_value(
        self, board: Board, move: Tuple[Tuple[int, int], Tuple[int, int]]
    ) -> float:
        """
        Gets the value of a capturing move.

        Args:
            board (Board): Current state of the chess board.
            move (Tuple[Tuple[int, int], Tuple[int, int]]): The move to be evaluated.

        Returns:
            capturing_value (float): The value of the capturing move.
        """

        target_pos = move[1]
        target_piece = board.get_piece(target_pos)

        piece_values = {
            "K": 0,  # King
            "Q": 9,  # Queen
            "R": 5,  # Rook
            "B": 3,  # Bishop
            "N": 3,  # Knight
            "P": 1,  # Pawn
        }

        if target_piece is not None and target_piece.color != board.active_color:
            # Move captures an enemy piece
            capturing_value = piece_values[str(target_piece).upper()]
        else:
            # Move does not capture a piece
            capturing_value = 0.0

        return capturing_value
