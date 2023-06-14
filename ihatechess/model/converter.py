from ..chess.board import Board

import numpy as np
import tensorflow as tf
import tf2onnx
from typing import List, Tuple


def encode_board(board: Board) -> np.ndarray:
    """
    Encode a chessboard state using 176 planes.

    :param board_state: 2D chessboard state
    :param repetitions: list of repetition counts for both players
    :return: 3D tensor with shape (8, 8, 176)
    """
    piece_mapping = {
        "K": 0,
        "Q": 1,
        "R": 2,
        "N": 3,
        "B": 4,
        "P": 5,
        "k": 6,
        "q": 7,
        "r": 8,
        "n": 9,
        "b": 10,
        "p": 11,
    }

    num_planes = 18
    chessboard = np.zeros((8, 8, num_planes), dtype=np.float32)

    for row in range(8):
        for col in range(8):
            piece = board.get_piece((row, col))
            if piece != None:
                plane_index = piece_mapping.get(str(piece), -1)
                chessboard[row, col, plane_index] = 1

    chessboard[:, :, 12] = int(board.castling_rights["white_kingside"])
    chessboard[:, :, 13] = int(board.castling_rights["white_queenside"])
    chessboard[:, :, 14] = int(board.castling_rights["black_kingside"])
    chessboard[:, :, 15] = int(board.castling_rights["black_queenside"])

    chessboard[:, :, 16] = board.fullmove_clock

    if board.en_passant_target != None:
        chessboard[*(board.en_passant_target), 17] = 1

    return chessboard


def encode_move(move: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[int, int, int]:
    """
    Encode a chess move into an index.

    :param board: The chessboard state.
    :param move: The move to encode as a tuple of tuples (initial_pos, final_pos).
    :return: The encoded index representing the move.
    """
    i, j = move[0]
    x, y = move[1]
    dx, dy = x - i, y - j
    if dx != 0 and dy == 0:  # north-south idx 0-13
        idx = 7 + dx
    elif dx == 0 and dy != 0:  # east-west idx 14-27
        idx = 21 + dy
    elif dx == dy:  # NW-SE idx 28-41
        idx = 35 + dx
    elif dx == -dy:  # NE-SW idx 42-55
        idx = 49 + dx
    elif (abs(dx) == 1 and abs(dy) == 2) or (
        abs(dx) == 2 and abs(dy) == 1
    ):  # Knight moves 56-63
        if (x, y) == (i + 2, j - 1):
            idx = 56
        elif (x, y) == (i + 2, j + 1):
            idx = 57
        elif (x, y) == (i + 1, j - 2):
            idx = 58
        elif (x, y) == (i - 1, j - 2):
            idx = 59
        elif (x, y) == (i - 2, j + 1):
            idx = 60
        elif (x, y) == (i - 2, j - 1):
            idx = 61
        elif (x, y) == (i - 1, j + 2):
            idx = 62
        elif (x, y) == (i + 1, j + 2):
            idx = 63
    else:
        raise ValueError("Invalid move!")

    return i, j, idx


def decode_move(
    policy_move: Tuple[int, int, int]
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Convert policy predictions to move in algebraic notation.

    :param policy: 8x8x64 policy predictions
    :return: move in algebraic notation
    """

    x, y, z = policy_move

    start = (x, y)

    if 0 <= z <= 13:
        dx = z - 7 if z < 7 else z - 6
        dy = 0
    elif 14 <= z <= 27:
        dy = z - 21 if z < 21 else z - 20
        dx = 0
    elif 28 <= z <= 41:
        dy = dx = z - 35 if z < 35 else z - 34
    elif 42 <= z <= 55:
        dx = z - 49 if z < 49 else z - 48
        dy = -dx
    elif 56 <= z <= 63:
        if z == 56:
            dx, dy = 2, -1
        elif z == 57:
            dx, dy = 2, 1
        elif z == 58:
            dx, dy = 1, -2
        elif z == 59:
            dx, dy = -1, -2
        elif z == 60:
            dx, dy = -2, 1
        elif z == 61:
            dx, dy = -2, -1
        elif z == 62:
            dx, dy = -1, 2
        else:  # z == 63
            dx, dy = 1, 2

    end = (x + dx, y + dy)

    return (start, end)


def convert_to_onnx(keras_model):
    input_tensor_spec = tf.TensorSpec([1, 8, 8, 18], keras_model.input.dtype)

    model_func = tf.function(lambda x: keras_model(x))

    onnx_model, _ = tf2onnx.convert.from_function(
        function=model_func,
        input_signature=[input_tensor_spec],
        opset=13,  # set the desired ONNX opset here
    )

    return onnx_model.SerializeToString()
