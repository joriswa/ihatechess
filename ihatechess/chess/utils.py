def position_to_algebraic(position):
    """
    Convert a position from a tuple of (row, column) to algebraic notation.
    """
    row, col = position
    return f"{chr(col + 97)}{8 - row}"


def algebraic_to_position(algebraic):
    """
    Convert a position from algebraic notation to a tuple of (row, column).
    """
    col = ord(algebraic[0]) - 97
    row = 8 - int(algebraic[1])
    return row, col


def move_to_uci(move):
    """
    Convert a move from algebraic notation to UCI notation.
    """
    start_pos = position_to_algebraic(move[0])
    end_pos = position_to_algebraic(move[1])
    return f"{start_pos}{end_pos}"


# def parse_fen(fen_str):
#     """Parses a FEN string into a chessboard."""
#     board, turn, castling, en_passant, halfmove, fullmove = fen_str.split(' ')

#     rows = board.split('/')
#     board = []
#     for row in rows:
#         board_row = []
#         for char in row:
#             if char.isdigit():
#                 for _ in range(int(char)):
#                     board_row.append(None)  # Empty squares
#             else:
#                 color = 'white' if char.isupper() else 'black'
#                 piece_type = char.lower()
#                 board_row.append(Piece.create(piece_type, color))  # Create the appropriate Piece instance
#         board.append(board_row)

#     # Return the board and other information
#     return {
#         'board': board,
#         'turn': turn,
#         'castling': castling,
#         'en_passant': en_passant,
#         'halfmove': halfmove,
#         'fullmove': fullmove,
#     }


def is_check(board, color):
    king_position = None
    for i in range(8):
        for j in range(8):
            piece = board[i][j]
            if isinstance(piece, King) and piece.color == color:
                king_position = (i, j)
                break
        if king_position is not None:
            break

    for i in range(8):
        for j in range(8):
            piece = board[i][j]
            if piece is not None and piece.color != color:
                if king_position in piece.get_valid_moves((i, j), board):
                    return True

    return False
