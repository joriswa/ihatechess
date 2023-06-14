from ihatechess.chess.utils import move_to_uci
from ihatechess.model.chessnet import ChessNet
from .mcts import MCTS
from ..chess.board import Board
from ..model.converter import convert_to_onnx, encode_board, encode_move
from multiprocessing import Manager, Pool

import coremltools as ct
from typing import List, Tuple
import numpy as np
from multiprocessing import current_process
import onnxruntime as rt


class DataGenerator:
    def __init__(
        self,
        inference_graph,
        num_games: int,
        num_simulations: int,
        noise_factor: int = 10,
        cpuct=1.0,
        temp_initial=1.0,
        temp_final=0.01,
        temp_threshold=30,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
    ):
        self.num_games = num_games
        self.num_simulations = num_simulations
        self.inference_graph = inference_graph
        self.noise_factor = noise_factor
        self.cpuct = cpuct
        self.temp_initial = temp_initial
        self.temp_final = temp_final
        self.temp_threshold = temp_threshold
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    def generate_single_game(self, inference_graph, num_simulations):
        worker_id = current_process()._identity[0]  # Get the unique worker ID
        print(f"Worker ID: {worker_id} started simulating a game...")
        net = rt.InferenceSession(inference_graph)
        mcts = MCTS(
            net,
            num_simulations,
            self.cpuct,
            self.temp_initial,
            self.temp_final,
            self.temp_threshold,
            self.dirichlet_alpha,
            self.dirichlet_eps,
        )  # Pass parameters to MCTS instance
        input_boards = []
        target_boards = []
        target_values = []

        board = Board()
        game_states = []
        game_moves = []
        legal_moves_masks = []
        game_turns = []

        while not board.is_game_over():
            legal_moves = board.generate_legal_moves()
            legal_moves_mask = np.zeros((8, 8, 64), dtype=np.bool8)
            for legal_move in legal_moves:
                legal_moves_mask[*encode_move(legal_move)] = True
            legal_moves_masks.append(legal_moves_mask)
            game_states.append(encode_board(board))

            # Use get_move to decide on the next move
            move = mcts.get_move(board)
            game_moves.append(move)
            game_turns.append(board.active_color)
            board.move_piece(*move)
            print(board)
            uci_move = move_to_uci(move)
            print(f"Worker ID: {worker_id} made move {uci_move}...")

        value = board.evaluate()

        for state, turn, move in zip(game_states, game_turns, game_moves):
            input_boards.append(state)
            target_board = np.zeros((8, 8, 64))
            target_board[*encode_move(move)] = 1
            target_boards.append(target_board)
            if (
                turn == "black"
            ):  # If it was black's turn in this state, switch the sign of the value
                value *= -1
            target_values.append(value)

        return input_boards, legal_moves_masks, target_boards, target_values

    def generate_games(self):
        print("Generating games...")
        with Pool() as pool:
            games = pool.starmap(
                self.generate_single_game,
                zip(
                    [self.inference_graph] * self.num_games,
                    [self.num_simulations] * self.num_games,
                ),
            )
        return [np.concatenate(game_data) for game_data in zip(*games)]
