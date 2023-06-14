from .pieces.king import King
from .pieces.queen import Queen
from .pieces.rook import Rook
from .pieces.bishop import Bishop
from .pieces.knight import Knight
from .pieces.pawn import Pawn

from .utils import algebraic_to_position, position_to_algebraic

from copy import copy
from IPython.display import SVG, display
from typing import List, Tuple, Self


class Board:
    """
    Represents the chess board. This includes the position of each piece as well as castling rights, en passante target, and the halfmove and fullmove clocks.
    """

    def __init__(self):
        self.board = self.create_board()
        self.en_passant_target = None
        self.castling_rights = {
            "white_kingside": True,
            "white_queenside": True,
            "black_kingside": True,
            "black_queenside": True,
        }
        self.repetitions = {"white": 0, "black": 0}
        self.halfmove_clock = 0
        self.fullmove_clock = 1
        self.active_color = "white"
        self.previous_boards = []
        self.legal_moves = []

    @staticmethod
    def create_board() -> Self:
        """
        Initializes a chess board in the starting position with white to move.

        :return: An object of type "Board" with repetition count, full- and half move, clock castling rights, and en passant target set to a starting chess game.
        """
        board = [[None for _ in range(8)] for _ in range(8)]

        for i in range(8):
            board[1][i] = Pawn("black")
        board[0] = [
            Rook("black"),
            Knight("black"),
            Bishop("black"),
            Queen("black"),
            King("black"),
            Bishop("black"),
            Knight("black"),
            Rook("black"),
        ]

        for i in range(8):
            board[6][i] = Pawn("white")
        board[7] = [
            Rook("white"),
            Knight("white"),
            Bishop("white"),
            Queen("white"),
            King("white"),
            Bishop("white"),
            Knight("white"),
            Rook("white"),
        ]

        return board

    def to_fen(self) -> str:
        """
        Converts the current board state to FE-Notation (Forsyth-Edwards-Notation).
        :return: FEN-String.
        """
        fen = ""

        # Encode the current permutation of pieces.
        for row in self.board:
            empty = 0
            for piece in row:
                if piece is None:
                    empty += 1
                else:
                    if empty > 0:
                        fen += str(empty)
                        empty = 0
                    fen += str(
                        piece
                    )  # The symbol method needs to be defined for each piece class
            if empty > 0:
                fen += str(empty)
            fen += "/"
        fen = fen[:-1]  # remove the last '/'

        # Encode the current active player.
        fen += " " + ("w" if self.active_color == "white" else "b") + " "

        if any(self.castling_rights.values()):
            if self.castling_rights["white_kingside"]:
                fen += "K"
            if self.castling_rights["white_queenside"]:
                fen += "Q"
            if self.castling_rights["black_kingside"]:
                fen += "k"
            if self.castling_rights["black_queenside"]:
                fen += "q"
        else:
            fen += "-"

        fen += " " + (
            position_to_algebraic(self.en_passant_target)
            if self.en_passant_target
            else "-"
        )

        # Relay the current half- and full move clock.
        fen += " " + str(self.halfmove_clock)
        fen += " " + str(self.fullmove_clock)

        return fen

    def load_fen(self, fen: str) -> Self:
        """
        Infers a board state in the sense of this class from given FEN.

        :return: A chess board based on the provided FEN.
        """
        self.board = [[None for _ in range(8)] for _ in range(8)]
        fen_parts = fen.split()

        # Parsing piece positions
        fen_board = fen_parts[0]
        row = 0  # Start from the first row instead of the last
        col = 0
        for char in fen_board:
            if char == "/":
                row += 1  # Increment row instead of decrementing
                col = 0
            elif char.isdigit():
                col += int(char)
            else:
                piece = None
                if char.lower() == "p":
                    piece = Pawn("white" if char.isupper() else "black")
                elif char.lower() == "r":
                    piece = Rook("white" if char.isupper() else "black")
                elif char.lower() == "n":
                    piece = Knight("white" if char.isupper() else "black")
                elif char.lower() == "b":
                    piece = Bishop("white" if char.isupper() else "black")
                elif char.lower() == "q":
                    piece = Queen("white" if char.isupper() else "black")
                elif char.lower() == "k":
                    piece = King("white" if char.isupper() else "black")

                if piece is not None:
                    self.board[row][col] = piece
                    col += 1

        self.active_color = "white" if fen_parts[1] == "w" else "black"

        # Parsing the castling rights
        castling_rights = fen_parts[2]
        castling_dict = {
            "K": "white_kingside",
            "Q": "white_queenside",
            "k": "black_kingside",
            "q": "black_queenside",
        }
        self.castling_rights = {
            "white_kingside": False,
            "white_queenside": False,
            "black_kingside": False,
            "black_queenside": False,
        }
        for char in castling_rights:
            if char in castling_dict:
                self.castling_rights[castling_dict[char]] = True

        # Parsing en passant target
        en_passant_target = fen_parts[3]
        if en_passant_target != "-":
            col, row = algebraic_to_position(en_passant_target)
            self.en_passant_target = (7 - row, col)  # Mirror on the vertical axis

        self.half_move_clock = int(fen_parts[4])  # Parsing the half-move clock
        self.full_move_clock = int(fen_parts[5])  # Parsing the full move number

    def __str__(self) -> str:
        """
        A string represenation of the board mainly used for debugging.

        :return: Human readable string encoding of the board.
        """
        output = ""
        for row in self.board:
            for piece in row:
                if piece is None:
                    output += ". "
                else:
                    output += str(piece) + " "
            output += "\n"
        return output

    def __eq__(self, other):
        """
        Evaluates if two chess positions are equivalent, based on the current piece positions and castling rights.

        :return: True if the given states are equivalent False otherwise.
        """
        if (
            self.active_color != other.active_color
            or self.castling_rights != other.castling_rights
        ):
            return False
        return all(
            str(self.board[i][j]) == str(other.board[i][j])
            for i in range(8)
            for j in range(8)
        )

    def clear(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]

    def set_piece(self, position, piece):
        self.board[position[0]][position[1]] = piece

    def get_piece(self, position):
        return self.board[position[0]][position[1]]

    def move_piece(self, start, end):
        sucessfully_moved = True
        capture_occured = False

        piece = self.get_piece(start)

        # Make sure the piece exists and is the correct color
        if piece is not None and piece.color == self.active_color:
            moves = piece.get_valid_moves(start, self.board, self.en_passant_target)

            # Make sure the move is valid for the selected piece
            if end in moves:
                temp_board = self.copy()

                temp_board.handle_castling(piece, (start, end))
                temp_board.update_castling_rights(start)

                capture_occured = temp_board.handle_en_passant_capture(
                    piece, (start, end)
                )

                # If the pawn has reached it's final row it's promoted to a queen,
                # otherwise just move the piece
                if not temp_board.handle_pawn_promotion(piece, (start, end)):
                    temp_board.board[end[0]][end[1]] = piece

                temp_board.board[start[0]][start[1]] = None
                piece.has_moved = True

                if temp_board.is_check(temp_board.active_color):
                    sucessfully_moved = False

                    return sucessfully_moved

                else:
                    # If the target square in the previous board wasn't empty, a capture has occured
                    if self.get_piece(end) is not None:
                        capture_occured = True

                    # It's the other player's turn now
                    # If it's blacks turn update the fullmove clock
                    temp_board.active_color = (
                        "white" if temp_board.active_color == "black" else "black"
                    )

                    temp_board.update_en_passant_target(piece, (start, end))

            else:
                sucessfully_moved = False

                return sucessfully_moved

        else:
            sucessfully_moved = False

            return sucessfully_moved

        if sucessfully_moved:
            self.legal_moves.clear()

            history = self.copy()
            history.previous_boards.clear()
            self.previous_boards.append(history)

            if not capture_occured:
                self.halfmove_clock += 1
            else:
                self.halfmove_clock = 0

            if temp_board.active_color == "black":
                self.fullmove_clock += 1

            self.board = temp_board.board
            self.castling_rights = temp_board.castling_rights
            self.active_color = temp_board.active_color
            self.en_passant_target = temp_board.en_passant_target

            self.legal_moves.clear()

            self.update_repetitions()

        return sucessfully_moved

    def is_stalemate(self):
        if self.is_check(self.active_color):
            return False

        legal_moves = self.generate_legal_moves()

        return len(legal_moves) == 0

    def update_repetitions(self):
        current_state = self.copy()
        current_state.previous_boards.clear()
        current_state.halfmove_clock = 0
        current_state.fullmove_clock = self.fullmove_clock

        repetitions = 0
        for previous_board in self.previous_boards:
            if previous_board == current_state:
                repetitions += 1
                if repetitions >= 2:  # The third repetition
                    color_that_just_moved = (
                        "white" if self.active_color == "black" else "black"
                    )
                    self.repetitions[color_that_just_moved] += 1
                    break

    def update_en_passant_target(self, piece, move):
        start, end = move

        if isinstance(piece, Pawn) and abs(start[0] - end[0]) == 2:
            if piece.color == "white":
                self.en_passant_target = ((start[0] + end[0]) // 2 - 1, start[1])
            else:
                self.en_passant_target = ((start[0] + end[0]) // 2 + 1, start[1])
        else:
            self.en_passant_target = None

    def handle_pawn_promotion(self, piece, move):
        _, end = move

        if isinstance(piece, Pawn) and end[0] in (0, 7):
            promoted_piece = Queen(piece.color)
            self.board[end[0]][end[1]] = promoted_piece
            return True

        return False

    def handle_en_passant_capture(self, piece, move):
        start, end = move

        if (
            isinstance(piece, Pawn)
            and start[1] != end[1]
            and self.get_piece(end) is None
        ):
            captured_pawn_position = (start[0], end[1])
            self.board[captured_pawn_position[0]][captured_pawn_position[1]] = None
            return True

        return False

    def handle_castling(self, piece, move):
        start, end = move

        if isinstance(piece, King) or isinstance(piece, Rook):
            if isinstance(piece, King) and abs(start[1] - end[1]) == 2:
                if end[1] > start[1]:  # Kingside castle
                    rook_start = (start[0], 7)
                    rook_end = (end[0], end[1] - 1)
                else:  # Queenside castle
                    rook_start = (start[0], 0)
                    rook_end = (end[0], end[1] + 1)

                rook = self.get_piece(rook_start)
                self.board[rook_end[0]][rook_end[1]] = rook
                self.board[rook_start[0]][rook_start[1]] = None
                rook.has_moved = True

    # only used to generate fen notation from board state
    def update_castling_rights(self, pos):
        piece = self.get_piece(pos)
        if isinstance(piece, King):
            self.castling_rights[f"{piece.color}_kingside"] = False
            self.castling_rights[f"{piece.color}_queenside"] = False
        elif isinstance(piece, Rook):
            if pos[1] == 0:  # Queenside rook
                self.castling_rights[f"{piece.color}_queenside"] = False
            elif pos[1] == 7:  # Kingside rook
                self.castling_rights[f"{piece.color}_kingside"] = False

    def is_check(self, color):
        king_position = None

        for i in range(8):
            for j in range(8):
                piece = self.get_piece((i, j))
                if isinstance(piece, King) and piece.color == color:
                    king_position = (i, j)
                    break

        if king_position is None:
            raise ValueError("No king found for the specified color")

        for i in range(8):
            for j in range(8):
                piece = self.get_piece((i, j))
                if piece is not None and piece.color != color:
                    moves = piece.get_valid_moves(
                        (i, j), self.board, self.en_passant_target
                    )
                    if king_position in moves:
                        return True

        return False

    def copy(self):
        copied_board = Board()  # Create a new instance of Board
        copied_board.board = [
            [None if piece is None else piece.copy() for piece in row]
            for row in self.board
        ]

        # Copy other attributes
        copied_board.en_passant_target = self.en_passant_target
        copied_board.castling_rights = self.castling_rights.copy()
        copied_board.active_color = self.active_color
        copied_board.repetitions = self.repetitions.copy()
        copied_board.halfmove_clock = self.halfmove_clock
        copied_board.fullmove_clock = self.fullmove_clock
        copied_board.previous_boards = copy(self.previous_boards)

        return copied_board

    def evaluate(self):
        if self.is_checkmate():
            if self.active_color == "white":
                return -1
            else:
                return 1
        elif (
            self.is_stalemate()
            or self.repetitions["white"] > 2
            or self.repetitions["black"] > 2
            or self.halfmove_clock >= 20
            or self.insufficient_material()
            or self.fullmove_clock >= 50
        ):
            return 0
        else:
            return None

    def generate_legal_moves(self):
        if self.legal_moves != []:
            return self.legal_moves

        legal_moves = []
        for row in range(8):
            for col in range(8):
                piece = self.get_piece((row, col))
                if piece is not None and piece.color == self.active_color:
                    moves = piece.get_valid_moves(
                        (row, col), self.board, self.en_passant_target
                    )
                    for move in moves:
                        temp_board = self.copy()
                        if not temp_board.move_piece((row, col), move):
                            continue

                        if not temp_board.is_check(self.active_color):
                            legal_moves.append(((row, col), move))

        self.legal_moves = legal_moves

        return legal_moves

    def is_checkmate(self):
        if not self.is_check(self.active_color):
            return False

        legal_moves = self.generate_legal_moves()

        return len(legal_moves) == 0

    def insufficient_material(self):
        piece_count = {
            "King": 0,
            "Queen": 0,
            "Rook": 0,
            "Bishop": 0,
            "Knight": 0,
            "Pawn": 0,
        }
        bishop_colors = {"white": 0, "black": 0}

        for i in range(8):
            for j in range(8):
                piece = self.get_piece((i, j))
                if piece is not None:
                    piece_count[type(piece).__name__] += 1
                    if isinstance(piece, Bishop):
                        if (i + j) % 2 == 0:  # light-squared bishop
                            bishop_colors["white"] += 1
                        else:  # dark-squared bishop
                            bishop_colors["black"] += 1

        # Conditions for insufficient material:
        # 1. Only kings left
        # 2. King and bishop(s) on same color
        # 3. King and knight(s)
        # Note: More than 1 bishop on different colors, more than 1 knight, or a pawn, rook or queen means the game can still end in checkmate.
        if (
            all(count == 0 for piece, count in piece_count.items() if piece != "King")
            or (
                piece_count["Bishop"] > 0
                and piece_count["Bishop"] == bishop_colors["white"]
                and all(
                    count == 0
                    for piece, count in piece_count.items()
                    if piece not in ["King", "Bishop"]
                )
            )
            or (
                piece_count["Bishop"] > 0
                and piece_count["Bishop"] == bishop_colors["black"]
                and all(
                    count == 0
                    for piece, count in piece_count.items()
                    if piece not in ["King", "Bishop"]
                )
            )
            or (
                piece_count["Knight"] > 0
                and all(
                    count == 0
                    for piece, count in piece_count.items()
                    if piece not in ["King", "Knight"]
                )
            )
        ):
            return True

        return False

    def is_game_over(self):
        return (
            self.is_checkmate()
            or self.is_stalemate()
            or self.repetitions["white"] > 2
            or self.repetitions["black"] > 2
            or self.halfmove_clock >= 20
            or self.insufficient_material()
            or self.fullmove_clock >= 50
        )

    def visualize(self):
        BOARD_SIZE = 8
        SQUARE_SIZE = 50  # size of each square on the board in pixels
        BOARD_COLOR_1 = "#DDB88C"
        BOARD_COLOR_2 = "#A66D4F"

        svg = '<svg width="{}" height="{}">'.format(
            BOARD_SIZE * SQUARE_SIZE, BOARD_SIZE * SQUARE_SIZE
        )

        # Draw squares
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                fill = BOARD_COLOR_1 if (i + j) % 2 == 0 else BOARD_COLOR_2
                svg += '<rect x="{}" y="{}" width="{}" height="{}" fill="{}" />'.format(
                    i * SQUARE_SIZE, j * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE, fill
                )

        # Draw pieces
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                piece = self.get_piece((j, i))
                if piece is not None:
                    piece_svg = piece.get_svg()
                    piece_svg = piece_svg.replace(
                        "<svg ",
                        '<svg x="{}" y="{}" '.format(
                            i * SQUARE_SIZE + 7, j * SQUARE_SIZE + 7
                        ),
                        1,
                    )
                    svg += piece_svg

        return display(SVG(svg))
