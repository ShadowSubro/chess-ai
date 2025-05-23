# "Best" AI logic + neural net

import chess
import torch
from network import ChessNet
from utils import board_to_tensor
import os
import random

class BestAI:
    def __init__(self, model_path=None, device="cpu"):
        self.model = ChessNet().to(device)
        self.device = device
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self,path):
        best_ai = BestAI(model_path="BestAI_model.pth")

    def select_move(self, board, exploration_prob=0.5, top_n=3):
        import torch
        from utils import board_to_tensor

        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 1:
            return legal_moves[0]
        move_values = []
        for move in legal_moves:
            board.push(move)
            tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, value = self.model(tensor)
            move_values.append((move, value.item()))
            board.pop()

        # For White, higher value is better; for Black, lower value is better
        reverse = board.turn == True  # True for White
        move_values.sort(key=lambda x: x[1], reverse=reverse)
        top_moves = move_values[:top_n] if len(move_values) >= top_n else move_values

        # Exploration: 20% random among top N, 80% best move
        if random.random() < exploration_prob:
            chosen_move = random.choice(top_moves)[0]
        else:
            chosen_move = top_moves[0][0]
        return chosen_move