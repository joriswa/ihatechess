from .piece import Piece

from typing import List, Tuple


class Knight(Piece):
    """
    Represents a knight chess piece. This class is a subclass of the Piece class.
    """

    def __str__(self) -> str:
        """
        Returns the string representation of the Knight.

        :return: 'N' if the Knight's color is white. Otherwise, returns 'n'.
        """
        return "N" if self.color == "white" else "n"

    def get_valid_moves(
        self,
        position: Tuple[int, int],
        board: List[List[Piece]],
        _en_passant_target: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Computes all legal moves for the current piece given a board state.

        :param position: A tuple representing the current position of the Knight.
        :param board: A 2D list of lists representing the chess board.
        :return: A list of tuples representing the valid moves.
        """
        valid_moves = []
        for i, j in [
            (2, 1),
            (1, 2),
            (-1, 2),
            (-2, 1),
            (-2, -1),
            (-1, -2),
            (1, -2),
            (2, -1),
        ]:  # Knight's moves
            new_position = (position[0] + i, position[1] + j)
            if 0 <= new_position[0] < 8 and 0 <= new_position[1] < 8:
                if (
                    board[new_position[0]][new_position[1]] is None
                    or board[new_position[0]][new_position[1]].color != self.color
                ):
                    valid_moves.append(new_position)
        return valid_moves

    def get_svg(self) -> str:
        """
        Returns the SVG representation of the Knight.

        :return: A string containing the SVG for the Knight.
        """
        svg_black = """
        <svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="45" height="45">
            <g style="opacity:1; fill:none; fill-opacity:1; fill-rule:evenodd; stroke:#000000; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:round; stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;" transform="translate(0,0.3)">
                <path
                d="M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18"
                style="fill:#000000; stroke:#000000;" />
                <path
                d="M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10"
                style="fill:#000000; stroke:#000000;" />
                <path
                d="M 9.5 25.5 A 0.5 0.5 0 1 1 8.5,25.5 A 0.5 0.5 0 1 1 9.5 25.5 z"
                style="fill:#ffffff; stroke:#ffffff;" />
                <path
                d="M 15 15.5 A 0.5 1.5 0 1 1  14,15.5 A 0.5 1.5 0 1 1  15 15.5 z"
                transform="matrix(0.866,0.5,-0.5,0.866,9.693,-5.173)"
                style="fill:#ffffff; stroke:#ffffff;" />
                <path
                d="M 24.55,10.4 L 24.1,11.85 L 24.6,12 C 27.75,13 30.25,14.49 32.5,18.75 C 34.75,23.01 35.75,29.06 35.25,39 L 35.2,39.5 L 37.45,39.5 L 37.5,39 C 38,28.94 36.62,22.15 34.25,17.66 C 31.88,13.17 28.46,11.02 25.06,10.5 L 24.55,10.4 z "
                style="fill:#ffffff; stroke:none;" />
            </g>
        </svg>
        """

        svg_white = """
        <svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="45" height="45">
            <g style="opacity:1; fill:none; fill-opacity:1; fill-rule:evenodd; stroke:#000000; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:round; stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;" transform="translate(0,0.3)">
                <path
                d="M 22,10 C 32.5,11 38.5,18 38,39 L 15,39 C 15,30 25,32.5 23,18"
                style="fill:#ffffff; stroke:#000000;" />
                <path
                d="M 24,18 C 24.38,20.91 18.45,25.37 16,27 C 13,29 13.18,31.34 11,31 C 9.958,30.06 12.41,27.96 11,28 C 10,28 11.19,29.23 10,30 C 9,30 5.997,31 6,26 C 6,24 12,14 12,14 C 12,14 13.89,12.1 14,10.5 C 13.27,9.506 13.5,8.5 13.5,7.5 C 14.5,6.5 16.5,10 16.5,10 L 18.5,10 C 18.5,10 19.28,8.008 21,7 C 22,7 22,10 22,10"
                style="fill:#ffffff; stroke:#000000;" />
                <path
                d="M 9.5 25.5 A 0.5 0.5 0 1 1 8.5,25.5 A 0.5 0.5 0 1 1 9.5 25.5 z"
                style="fill:#000000; stroke:#000000;" />
                <path
                d="M 15 15.5 A 0.5 1.5 0 1 1  14,15.5 A 0.5 1.5 0 1 1  15 15.5 z"
                transform="matrix(0.866,0.5,-0.5,0.866,9.693,-5.173)"
                style="fill:#000000; stroke:#000000;" />
            </g>
        </svg>
        """

        return svg_white if self.color == "white" else svg_black
