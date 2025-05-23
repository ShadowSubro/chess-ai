# "Destroyer" AI logic + neural net

import random
import torch
import chess
from network import ChessNet
from utils import board_to_tensor

class DestroyerAI:
    def __init__(self, model_path=None, device="cpu"):
        self.model = ChessNet().to(device)
        self.device = device
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        pass

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self,path):
        destroyer_ai = DestroyerAI(model_path="DestroyerAI_model.pth")

    def select_move(self, board, exploration_prob=0.5, top_n=3):
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 1:
            return legal_moves[0]
        move_values = []
        for move in legal_moves:
            board.push(move)
            tensor = board_to_tensor(board).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, value = self.model(tensor)
            # Aggressive: bonus for captures and checks
            bonus = 0
            if board.is_check():
                bonus += 0.3
            if board.is_capture(move):
                bonus += 0.5
            score = value.item() + bonus
            move_values.append((move, score))
            board.pop()
        # For Black, lower value is better; for White, higher value is better
        reverse = board.turn == chess.WHITE
        move_values.sort(key=lambda x: x[1], reverse=reverse)
        top_moves = move_values[:top_n] if len(move_values) >= top_n else move_values
        # Exploration: 20% random among top N, 80% best move
        if random.random() < exploration_prob:
            chosen_move = random.choice(top_moves)[0]
        else:
            chosen_move = top_moves[0][0]
        return chosen_move