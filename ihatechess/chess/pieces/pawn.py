from .piece import Piece

from typing import List, Tuple


class Pawn(Piece):
    """
    Represents a pawn chess piece. This class is a subclass of the Piece class.
    """

    def __str__(self) -> str:
        """
        Returns the string representation of the Pawn.

        :return: 'P' if the Pawn's color is white. Otherwise, returns 'p'.
        """
        return "P" if self.color == "white" else "p"

    def get_valid_moves(
        self,
        position: Tuple[int, int],
        board: List[List[Piece]],
        en_passant_target: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Computes all legal moves for the current piece given a board state.

        :param position: A tuple representing the current position of the Pawn.
        :param board: A 2D list representing the chess board.
        :param en_passant_target: A tuple representing the position that can be attacked en passant.
        :return: A list of tuples representing the valid moves.
        """
        valid_moves = []
        direction = -1 if self.color == "white" else 1
        start_row = 6 if self.color == "white" else 1

        # If the square in front of the pawn is empty, it can move there
        new_position = (position[0] + direction, position[1])
        if 0 <= new_position[0] < 8 and board[new_position[0]][new_position[1]] is None:
            valid_moves.append(new_position)

        # If the pawn hasn't moved yet, it can move two squares forward if both squares are empty
        if position[0] == start_row:
            new_position = (position[0] + 2 * direction, position[1])
            if (
                board[new_position[0]][new_position[1]] is None
                and board[new_position[0] - direction][new_position[1]] is None
            ):
                valid_moves.append(new_position)

        # If the pawn can capture a piece diagonally, it can move there
        for j in [-1, 1]:
            new_position = (position[0] + direction, position[1] + j)
            if (
                0 <= new_position[0] < 8
                and 0 <= new_position[1] < 8
                and board[new_position[0]][new_position[1]] is not None
                and board[new_position[0]][new_position[1]].color != self.color
            ):
                valid_moves.append(new_position)

        # Handle en passant
        if en_passant_target is not None:
            if self.color == "white" and position[0] == 3:
                if (
                    position[1] - 1 == en_passant_target[1]
                    or position[1] + 1 == en_passant_target[1]
                ):
                    valid_moves.append((en_passant_target[0] - 1, en_passant_target[1]))
            elif self.color == "black" and position[0] == 4:
                if (
                    position[1] - 1 == en_passant_target[1]
                    or position[1] + 1 == en_passant_target[1]
                ):
                    valid_moves.append((en_passant_target[0] + 1, en_passant_target[1]))

        return valid_moves

    def get_svg(self) -> str:
        """
        Returns the SVG representation of the Pawn.

        :return: A string containing the SVG for the Pawn.
        """
        svg_black = """
        <svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="45" height="45">
            <path d="m 22.5,9 c -2.21,0 -4,1.79 -4,4 0,0.89 0.29,1.71 0.78,2.38 C 17.33,16.5 16,18.59 16,21 c 0,2.03 0.94,3.84 2.41,5.03 C 15.41,27.09 11,31.58 11,39.5 H 34 C 34,31.58 29.59,27.09 26.59,26.03 28.06,24.84 29,23.03 29,21 29,18.59 27.67,16.5 25.72,15.38 26.21,14.71 26.5,13.89 26.5,13 c 0,-2.21 -1.79,-4 -4,-4 z" style="opacity:1; fill:#000000; fill-opacity:1; fill-rule:nonzero; stroke:#000000; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:miter; stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;"/>
        </svg>
        """
        svg_white = """
        <svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="45" height="45">
            <path d="m 22.5,9 c -2.21,0 -4,1.79 -4,4 0,0.89 0.29,1.71 0.78,2.38 C 17.33,16.5 16,18.59 16,21 c 0,2.03 0.94,3.84 2.41,5.03 C 15.41,27.09 11,31.58 11,39.5 H 34 C 34,31.58 29.59,27.09 26.59,26.03 28.06,24.84 29,23.03 29,21 29,18.59 27.67,16.5 25.72,15.38 26.21,14.71 26.5,13.89 26.5,13 c 0,-2.21 -1.79,-4 -4,-4 z" style="opacity:1; fill:#ffffff; fill-opacity:1; fill-rule:nonzero; stroke:#000000; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:miter; stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;"/>
        </svg>
        """

        return svg_white if self.color == "white" else svg_black
