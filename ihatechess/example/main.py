from ihatechess.chess.board import Board

import pygame
import sys
import io
import cairosvg
from ihatechess.chess.pieces.pawn import Pawn
from ihatechess.chess.pieces.queen import Queen
from ihatechess.chess.pieces.rook import Rook
from ihatechess.chess.pieces.king import King
from ihatechess.chess.pieces.knight import Knight
from ihatechess.chess.pieces.bishop import Bishop
from ihatechess.chess.pieces.piece import Piece

# Here, we initialize pygame
pygame.init()

# We can set the dimensions of the chessboard
WINDOW_DIMENSION = 800
WINDOW = pygame.display.set_mode((WINDOW_DIMENSION, WINDOW_DIMENSION))
BOARD_COLOR_1 = "#DDB88C"
BOARD_COLOR_2 = "#A66D4F"

# Colors we may use for drawing the chess board and pieces
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
BLUE = (0, 0, 255)

# Dictionary to store piece images
piece_images = {}


def draw_highlight(surface, color, rect, alpha):
    temp_surface = pygame.Surface((100, 100))  # Create a temporary surface
    temp_surface.set_alpha(
        alpha
    )  # Set alpha level. The value must be between 0 (completely transparent) and 255 (completely opaque).
    temp_surface.fill(color)  # Fill it with your color
    surface.blit(
        temp_surface, (rect[0] * 100, rect[1] * 100)
    )  # Blit it onto the original surface


def load_svg(svg_string, scale=1):
    svg_data = svg_string.encode()
    png_data = cairosvg.svg2png(bytestring=svg_data, scale=scale)
    png_byte_io = io.BytesIO(png_data)
    return pygame.image.load(png_byte_io, "png_byte_file.png")


# Function to load all piece images
def load_images():
    pieces = [
        Pawn("white"),
        Rook("white"),
        Knight("white"),
        Bishop("white"),
        Queen("white"),
        King("white"),
        Pawn("black"),
        Rook("black"),
        Knight("black"),
        Bishop("black"),
        Queen("black"),
        King("black"),
    ]

    for piece in pieces:
        image = load_svg(piece.get_svg(), scale=2)
        piece_images[str(piece)] = image


def draw_board(board, legal_moves=[]):
    for row in range(8):
        for col in range(8):
            color = BOARD_COLOR_1 if (row + col) % 2 == 0 else BOARD_COLOR_2
            # If the square is a legal move, change its color
            pygame.draw.rect(WINDOW, color, (col * 100, row * 100, 100, 100))
            if (row, col) in legal_moves:
                draw_highlight(WINDOW, GREEN, (col, row), 128)
            piece = board.get_piece((row, col))
            if piece is not None:
                image = piece_images[str(piece)]
                WINDOW.blit(image, (col * 100 + 3, row * 100))


# The main function
def main():
    load_images()
    board = Board()
    legal_moves = []
    selected_piece = None
    selected_piece_position = None
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                position = pygame.mouse.get_pos()
                col, row = position[0] // 100, position[1] // 100
                if selected_piece is None:
                    selected_piece = board.get_piece((row, col))
                    if selected_piece is not None:
                        selected_piece_position = (row, col)
                        legal_moves = selected_piece.get_valid_moves(
                            (row, col), board.board, board.en_passant_target
                        )
                else:
                    if (row, col) in legal_moves:
                        board.move_piece(selected_piece_position, (row, col))
                    selected_piece = None
                    legal_moves = []

        draw_board(board, legal_moves)
        pygame.display.update()


# Run the game
if __name__ == "__main__":
    main()
