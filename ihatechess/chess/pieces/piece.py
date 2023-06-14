from typing import List, Self, Tuple


class Piece:
    """
    This class represents a generic chess piece. This class is a superclass of the King, Queen, Rook, Bishop, Knight, and Pawn classes.
    """

    def __init__(self, color: str):
        """
        :param color: A string representing the color of the piece. Either 'white' or 'black'.
        """
        self.color = color
        self.has_moved = False

    def get_valid_moves(
        self,
        _position: Tuple[int, int],
        _board: List[List[Self]],
        _en_passante_target: Tuple[int, int],
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Abstract method that returns the valid moves for the piece.
        """
        raise NotImplementedError("This method should be implemented by each subclass")

    def get_svg(self) -> str:
        """
        Abstract method that returns the SVG representation of the piece.
        """
        raise NotImplementedError("This method should be implemented by each subclass")

    def copy(self) -> Self:
        """
        Creates a copy of the piece.
        """
        copied_piece = type(self)(self.color)
        copied_piece.has_moved = self.has_moved
        return copied_piece
