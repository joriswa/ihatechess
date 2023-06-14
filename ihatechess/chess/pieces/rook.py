from .piece import Piece

from typing import List, Tuple


class Rook(Piece):
    def __str__(self) -> str:
        """
        Returns the string representation of the Rook. Upper case for white lower case for black.
        """
        return "R" if self.color == "white" else "r"

    def get_valid_moves(
        self,
        position: Tuple[int, int],
        board: List[List[Piece]],
        _en_passante_target: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Computes all legal moves for the current piece given a board state.

        :param position: A tuple representing the current position of the Rook.
        :param board: A 2D list representing the chess board.
        :return: A list of tuples representing the valid moves.
        """
        valid_moves = []
        for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
            for k in range(1, 8):
                new_position = (position[0] + i * k, position[1] + j * k)
                if 0 <= new_position[0] < 8 and 0 <= new_position[1] < 8:
                    if board[new_position[0]][new_position[1]] is None:
                        valid_moves.append(new_position)
                    elif board[new_position[0]][new_position[1]].color != self.color:
                        valid_moves.append(new_position)
                        break
                    else:
                        break
        return valid_moves

    def get_svg(self) -> str:
        """
        :return: A string containing the SVG for the Rook.
        """
        svg_black = """
        <svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="45" height="45">
        <g style="opacity:1; fill:#000000; fill-opacity:1; fill-rule:evenodd; stroke:#000000; stroke-width:1.5; stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;" transform="translate(0,0.3)">
            <path
            d="M 9,39 L 36,39 L 36,36 L 9,36 L 9,39 z "
            style="stroke-linecap:butt;" />
            <path
            d="M 12.5,32 L 14,29.5 L 31,29.5 L 32.5,32 L 12.5,32 z "
            style="stroke-linecap:butt;" />
            <path
            d="M 12,36 L 12,32 L 33,32 L 33,36 L 12,36 z "
            style="stroke-linecap:butt;" />
            <path
            d="M 14,29.5 L 14,16.5 L 31,16.5 L 31,29.5 L 14,29.5 z "
            style="stroke-linecap:butt;stroke-linejoin:miter;" />
            <path
            d="M 14,16.5 L 11,14 L 34,14 L 31,16.5 L 14,16.5 z "
            style="stroke-linecap:butt;" />
            <path
            d="M 11,14 L 11,9 L 15,9 L 15,11 L 20,11 L 20,9 L 25,9 L 25,11 L 30,11 L 30,9 L 34,9 L 34,14 L 11,14 z "
            style="stroke-linecap:butt;" />
            <path
            d="M 12,35.5 L 33,35.5 L 33,35.5"
            style="fill:none; stroke:#ffffff; stroke-width:1; stroke-linejoin:miter;" />
            <path
            d="M 13,31.5 L 32,31.5"
            style="fill:none; stroke:#ffffff; stroke-width:1; stroke-linejoin:miter;" />
            <path
            d="M 14,29.5 L 31,29.5"
            style="fill:none; stroke:#ffffff; stroke-width:1; stroke-linejoin:miter;" />
            <path
            d="M 14,16.5 L 31,16.5"
            style="fill:none; stroke:#ffffff; stroke-width:1; stroke-linejoin:miter;" />
            <path
            d="M 11,14 L 34,14"
            style="fill:none; stroke:#ffffff; stroke-width:1; stroke-linejoin:miter;" />
        </g>
        </svg>
        """

        svg_white = """
        <svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="45" height="45">
        <g style="opacity:1; fill:#ffffff; fill-opacity:1; fill-rule:evenodd; stroke:#000000; stroke-width:1.5; stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;" transform="translate(0,0.3)">
            <path
            d="M 9,39 L 36,39 L 36,36 L 9,36 L 9,39 z "
            style="stroke-linecap:butt;" />
            <path
            d="M 12,36 L 12,32 L 33,32 L 33,36 L 12,36 z "
            style="stroke-linecap:butt;" />
            <path
            d="M 11,14 L 11,9 L 15,9 L 15,11 L 20,11 L 20,9 L 25,9 L 25,11 L 30,11 L 30,9 L 34,9 L 34,14"
            style="stroke-linecap:butt;" />
            <path
            d="M 34,14 L 31,17 L 14,17 L 11,14" />
            <path
            d="M 31,17 L 31,29.5 L 14,29.5 L 14,17"
            style="stroke-linecap:butt; stroke-linejoin:miter;" />
            <path
            d="M 31,29.5 L 32.5,32 L 12.5,32 L 14,29.5" />
            <path
            d="M 11,14 L 34,14"
            style="fill:none; stroke:#000000; stroke-linejoin:miter;" />
        </g>
        </svg>
        """

        return svg_white if self.color == "white" else svg_black
