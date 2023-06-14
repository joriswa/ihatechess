from .piece import Piece

from typing import List, Tuple


class Queen(Piece):
    def __str__(self) -> str:
        """
        :return: String representation of the Queen. Upper case for white lower case for black.
        """
        return "Q" if self.color == "white" else "q"

    def get_valid_moves(
        self,
        position: Tuple[int, int],
        board: List[List[Piece]],
        _en_passant_target: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Computes all legal moves for the current piece given a board state.

        :param position: A tuple representing the current position of the Queen.
        :param board: A 2D list representing the chess board.
        :return: A list of tuples representing the valid moves.
        """
        valid_moves = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i != 0 or j != 0:
                    for k in range(1, 8):
                        new_position = (position[0] + i * k, position[1] + j * k)
                        if 0 <= new_position[0] < 8 and 0 <= new_position[1] < 8:
                            if board[new_position[0]][new_position[1]] is None:
                                valid_moves.append(new_position)
                            elif (
                                board[new_position[0]][new_position[1]].color
                                != self.color
                            ):
                                valid_moves.append(new_position)
                                break
                            else:
                                break
        return valid_moves

    def get_svg(self) -> str:
        """
        :return: A string containing the SVG for the Queen.
        """
        svg_black = """
        <svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="45" height="45">
        <g style="fill:#000000;stroke:#000000;stroke-width:1.5; stroke-linecap:round;stroke-linejoin:round">

            <path d="M 9,26 C 17.5,24.5 30,24.5 36,26 L 38.5,13.5 L 31,25 L 30.7,10.9 L 25.5,24.5 L 22.5,10 L 19.5,24.5 L 14.3,10.9 L 14,25 L 6.5,13.5 L 9,26 z"
            style="stroke-linecap:butt;fill:#000000" />
            <path d="m 9,26 c 0,2 1.5,2 2.5,4 1,1.5 1,1 0.5,3.5 -1.5,1 -1,2.5 -1,2.5 -1.5,1.5 0,2.5 0,2.5 6.5,1 16.5,1 23,0 0,0 1.5,-1 0,-2.5 0,0 0.5,-1.5 -1,-2.5 -0.5,-2.5 -0.5,-2 0.5,-3.5 1,-2 2.5,-2 2.5,-4 -8.5,-1.5 -18.5,-1.5 -27,0 z" />
            <path d="M 11.5,30 C 15,29 30,29 33.5,30" />
            <path d="m 12,33.5 c 6,-1 15,-1 21,0" />
            <circle cx="6" cy="12" r="2" />
            <circle cx="14" cy="9" r="2" />
            <circle cx="22.5" cy="8" r="2" />
            <circle cx="31" cy="9" r="2" />
            <circle cx="39" cy="12" r="2" />
            <path d="M 11,38.5 A 35,35 1 0 0 34,38.5"
            style="fill:none; stroke:#000000;stroke-linecap:butt;" />
            <g style="fill:none; stroke:#ffffff;">
            <path d="M 11,29 A 35,35 1 0 1 34,29" />
            <path d="M 12.5,31.5 L 32.5,31.5" />
            <path d="M 11.5,34.5 A 35,35 1 0 0 33.5,34.5" />
            <path d="M 10.5,37.5 A 35,35 1 0 0 34.5,37.5" />
            </g>
        </g>
        </svg>
        """

        svg_white = """
        <svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="45" height="45">
        <g style="fill:#ffffff;stroke:#000000;stroke-width:1.5;stroke-linejoin:round">
            <path d="M 9,26 C 17.5,24.5 30,24.5 36,26 L 38.5,13.5 L 31,25 L 30.7,10.9 L 25.5,24.5 L 22.5,10 L 19.5,24.5 L 14.3,10.9 L 14,25 L 6.5,13.5 L 9,26 z"/>
            <path d="M 9,26 C 9,28 10.5,28 11.5,30 C 12.5,31.5 12.5,31 12,33.5 C 10.5,34.5 11,36 11,36 C 9.5,37.5 11,38.5 11,38.5 C 17.5,39.5 27.5,39.5 34,38.5 C 34,38.5 35.5,37.5 34,36 C 34,36 34.5,34.5 33,33.5 C 32.5,31 32.5,31.5 33.5,30 C 34.5,28 36,28 36,26 C 27.5,24.5 17.5,24.5 9,26 z"/>
            <path d="M 11.5,30 C 15,29 30,29 33.5,30" style="fill:none"/>
            <path d="M 12,33.5 C 18,32.5 27,32.5 33,33.5" style="fill:none"/>
            <circle cx="6" cy="12" r="2" />
            <circle cx="14" cy="9" r="2" />
            <circle cx="22.5" cy="8" r="2" />
            <circle cx="31" cy="9" r="2" />
            <circle cx="39" cy="12" r="2" />
        </g>
        </svg>
        """

        return svg_white if self.color == "white" else svg_black
