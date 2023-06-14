from .piece import Piece
from .rook import Rook
from .pawn import Pawn

from typing import List, Tuple


class King(Piece):
    """
    This class represents a king chess piece. This class is a subclass of the Piece class.
    """

    def __str__(self) -> str:
        """
        :return: String representation of the King. Upper case for white lower case for black.
        """
        return "K" if self.color == "white" else "k"

    def get_valid_moves(
        self,
        position: Tuple[int, int],
        board: List[List[Piece]],
        en_passant_target: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        :param position: A tuple representing the current position of the King.
        :param board: A 2D list representing the chess board.
        :param en_passant_target: A tuple representing the position that can be attacked en passant.
        :return: A list of tuples representing the valid moves.
        """

        valid_moves = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                new_position = (position[0] + i, position[1] + j)
                if 0 <= new_position[0] < 8 and 0 <= new_position[1] < 8:
                    if (
                        board[new_position[0]][new_position[1]] is None
                        or board[new_position[0]][new_position[1]].color != self.color
                    ):
                        valid_moves.append(new_position)
        if not self.has_moved:
            # King side castling
            if (
                isinstance(board[position[0]][position[1] + 3], Rook)
                and not isinstance(board[position[0]][position[1] + 3], King)
                and not isinstance(board[position[0]][position[1] + 3], Pawn)
                and not board[position[0]][position[1] + 3].has_moved
                and board[position[0]][position[1] + 1] is None
                and board[position[0]][position[1] + 2] is None
            ):
                # Check if moving to the castling square results in a check
                if not self.is_check_after_move(
                    position, (position[0], position[1] + 2), board, en_passant_target
                ):
                    valid_moves.append((position[0], position[1] + 2))
            # Queen side castling
            if (
                isinstance(board[position[0]][position[1] - 4], Rook)
                and not isinstance(board[position[0]][position[1] - 4], King)
                and not isinstance(board[position[0]][position[1] - 4], Pawn)
                and not board[position[0]][position[1] - 4].has_moved
                and board[position[0]][position[1] - 1] is None
                and board[position[0]][position[1] - 2] is None
                and board[position[0]][position[1] - 3] is None
            ):
                # Check if moving to the castling square results in a check
                if not self.is_check_after_move(
                    position, (position[0], position[1] - 2), board, en_passant_target
                ):
                    valid_moves.append((position[0], position[1] - 2))

        return valid_moves

    def is_check_after_move(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        board: List[List[Piece]],
        en_passant_target: Tuple[int, int],
    ) -> bool:
        """
        :param start: A tuple representing the current position of the King.
        :param end: A tuple representing the position that the King is moving to.
        :param board: A 2D list representing the chess board.
        :param en_passant_target: A tuple representing the position that can be attacked en passant.
        """
        temp_board = [
            [None if piece is None else piece.copy() for piece in row] for row in board
        ]

        # Make the move on the temporary board
        temp_board[end[0]][end[1]] = temp_board[start[0]][start[1]]
        temp_board[start[0]][start[1]] = None

        temp_board[end[0]][end[1]].has_moved = True

        # Handle castling
        if isinstance(temp_board[end[0]][end[1]], King) and abs(start[1] - end[1]) == 2:
            if end[1] > start[1]:  # Kingside castle
                rook_start = (start[0], 7)
                rook_end = (end[0], end[1] - 1)
            else:  # Queenside castle
                rook_start = (start[0], 0)
                rook_end = (end[0], end[1] + 1)

            temp_board[rook_end[0]][rook_end[1]] = temp_board[rook_start[0]][
                rook_start[1]
            ]
            temp_board[rook_start[0]][rook_start[1]] = None

            temp_board[rook_end[0]][rook_end[1]].has_moved = True

        # Retrieve king position
        king_position = None
        for i in range(8):
            for j in range(8):
                piece = temp_board[i][j]
                if isinstance(piece, King) and piece.color == self.color:
                    king_position = (i, j)
                    break

        # Check if the King is in check on the temporary board
        for i in range(8):
            for j in range(8):
                piece = temp_board[i][j]
                if piece is not None and piece.color != self.color:
                    moves = piece.get_valid_moves((i, j), temp_board, en_passant_target)
                    if king_position in moves:
                        return True

        return False

    def get_svg(self) -> str:
        """
        :return: A string containing the SVG for the King.
        """
        svg_black = """
        <svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="45" height="45">
        <g style="fill:none; fill-opacity:1; fill-rule:evenodd; stroke:#000000; stroke-width:1.5; stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;">
            <path d="M 22.5,11.63 L 22.5,6" style="fill:none; stroke:#000000; stroke-linejoin:miter;" id="path6570"/>
            <path d="M 22.5,25 C 22.5,25 27,17.5 25.5,14.5 C 25.5,14.5 24.5,12 22.5,12 C 20.5,12 19.5,14.5 19.5,14.5 C 18,17.5 22.5,25 22.5,25" style="fill:#000000;fill-opacity:1; stroke-linecap:butt; stroke-linejoin:miter;"/>
            <path d="M 12.5,37 C 18,40.5 27,40.5 32.5,37 L 32.5,30 C 32.5,30 41.5,25.5 38.5,19.5 C 34.5,13 25,16 22.5,23.5 L 22.5,27 L 22.5,23.5 C 20,16 10.5,13 6.5,19.5 C 3.5,25.5 12.5,30 12.5,30 L 12.5,37" style="fill:#000000; stroke:#000000;"/>
            <path d="M 20,8 L 25,8" style="fill:none; stroke:#000000; stroke-linejoin:miter;"/>
            <path d="M 32,29.5 C 32,29.5 40.5,25.5 38.03,19.85 C 34.15,14 25,18 22.5,24.5 L 22.5,26.6 L 22.5,24.5 C 20,18 10.85,14 6.97,19.85 C 4.5,25.5 13,29.5 13,29.5" style="fill:none; stroke:#ffffff;"/>
            <path d="M 12.5,30 C 18,27 27,27 32.5,30 M 12.5,33.5 C 18,30.5 27,30.5 32.5,33.5 M 12.5,37 C 18,34 27,34 32.5,37" style="fill:none; stroke:#ffffff;"/>
        </g>
        </svg>
        """
        svg_white = """
        <svg xmlns="http://www.w3.org/2000/svg" width="45" height="45">
        <g fill="none" fill-rule="evenodd" stroke="#000" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5">
            <path stroke-linejoin="miter" d="M22.5 11.63V6M20 8h5"/>
            <path fill="#fff" stroke-linecap="butt" stroke-linejoin="miter" d="M22.5 25s4.5-7.5 3-10.5c0 0-1-2.5-3-2.5s-3 2.5-3 2.5c-1.5 3 3 10.5 3 10.5"/>
            <path fill="#fff" d="M12.5 37c5.5 3.5 14.5 3.5 20 0v-7s9-4.5 6-10.5c-4-6.5-13.5-3.5-16 4V27v-3.5c-2.5-7.5-12-10.5-16-4-3 6 6 10.5 6 10.5v7"/>
            <path d="M12.5 30c5.5-3 14.5-3 20 0m-20 3.5c5.5-3 14.5-3 20 0m-20 3.5c5.5-3 14.5-3 20 0"/>
        </g>
        </svg>
        """

        return svg_white if self.color == "white" else svg_black
