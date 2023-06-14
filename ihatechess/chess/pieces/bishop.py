from .piece import Piece

from typing import List, Tuple


class Bishop(Piece):
    """
    This class represents a bishop chess piece. This class is a subclass of the Piece class.
    """

    def __str__(self) -> str:
        """
        :return: String representation of the Bishop. Upper case for white lower case for black.
        """
        return "B" if self.color == "white" else "b"

    def get_valid_moves(self, position: Tuple[int, int], board: List[List[Piece]], _):
        """
        :param position: A tuple representing the current position of the Bishop.
        :param board: A 2D list representing the chess board.
        :return: A list of tuples representing the valid moves.
        """
        valid_moves = []
        for i, j in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
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
        :return: A string containing the SVG for the Bishop.
        """
        svg_black = """
        <svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="45" height="45">
        <g style="opacity:1; fill:none; fill-rule:evenodd; fill-opacity:1; stroke:#000000; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:round; stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;" transform="translate(0,0.6)">
            <g style="fill:#000000; stroke:#000000; stroke-linecap:butt;">
            <path d="M 9,36 C 12.39,35.03 19.11,36.43 22.5,34 C 25.89,36.43 32.61,35.03 36,36 C 36,36 37.65,36.54 39,38 C 38.32,38.97 37.35,38.99 36,38.5 C 32.61,37.53 25.89,38.96 22.5,37.5 C 19.11,38.96 12.39,37.53 9,38.5 C 7.65,38.99 6.68,38.97 6,38 C 7.35,36.54 9,36 9,36 z"/>
            <path d="M 15,32 C 17.5,34.5 27.5,34.5 30,32 C 30.5,30.5 30,30 30,30 C 30,27.5 27.5,26 27.5,26 C 33,24.5 33.5,14.5 22.5,10.5 C 11.5,14.5 12,24.5 17.5,26 C 17.5,26 15,27.5 15,30 C 15,30 14.5,30.5 15,32 z"/>
            <path d="M 25 8 A 2.5 2.5 0 1 1  20,8 A 2.5 2.5 0 1 1  25 8 z"/>
            </g>
            <path d="M 17.5,26 L 27.5,26 M 15,30 L 30,30 M 22.5,15.5 L 22.5,20.5 M 20,18 L 25,18" style="fill:none; stroke:#ffffff; stroke-linejoin:miter;"/>
        </g>
        </svg>
        """

        svg_white = """
        <svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="45" height="45">
        <g style="opacity:1; fill:none; fill-rule:evenodd; fill-opacity:1; stroke:#000000; stroke-width:1.5; stroke-linecap:round; stroke-linejoin:round; stroke-miterlimit:4; stroke-dasharray:none; stroke-opacity:1;" transform="translate(0,0.6)">
            <g style="fill:#ffffff; stroke:#000000; stroke-linecap:butt;">
            <path d="M 9,36 C 12.39,35.03 19.11,36.43 22.5,34 C 25.89,36.43 32.61,35.03 36,36 C 36,36 37.65,36.54 39,38 C 38.32,38.97 37.35,38.99 36,38.5 C 32.61,37.53 25.89,38.96 22.5,37.5 C 19.11,38.96 12.39,37.53 9,38.5 C 7.65,38.99 6.68,38.97 6,38 C 7.35,36.54 9,36 9,36 z"/>
            <path d="M 15,32 C 17.5,34.5 27.5,34.5 30,32 C 30.5,30.5 30,30 30,30 C 30,27.5 27.5,26 27.5,26 C 33,24.5 33.5,14.5 22.5,10.5 C 11.5,14.5 12,24.5 17.5,26 C 17.5,26 15,27.5 15,30 C 15,30 14.5,30.5 15,32 z"/>
            <path d="M 25 8 A 2.5 2.5 0 1 1  20,8 A 2.5 2.5 0 1 1  25 8 z"/>
            </g>
            <path d="M 17.5,26 L 27.5,26 M 15,30 L 30,30 M 22.5,15.5 L 22.5,20.5 M 20,18 L 25,18" style="fill:none; stroke:#000000; stroke-linejoin:miter;"/>
        </g>
        </svg>
        """

        return svg_white if self.color == "white" else svg_black
