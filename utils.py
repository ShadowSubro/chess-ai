# Where logs and models are saved

# Utility functions: board to tensor, move encoding, saving/loading, etc.

import chess
import numpy as np
import torch

def board_to_tensor(board):
    """
    Converts a python-chess Board into a (12, 8, 8) tensor.
    12 planes: [wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK]
    """
    piece_map = board.piece_map()
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_to_plane = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    for square, piece in piece_map.items():
        row = 7 - (square // 8)
        col = square % 8
        plane = piece_to_plane[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
        tensor[plane, row, col] = 1
    return torch.tensor(tensor)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model