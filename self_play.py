# Code for self-play loop

import chess
from ai_best import BestAI
from ai_destroyer import DestroyerAI
from utils import board_to_tensor

def self_play_game():
    board = chess.Board()
    ai_white = BestAI(model_path="BestAI_model.pth")
    ai_black = DestroyerAI(model_path="DestroyerAI_model.pth")
    move_history = []
    training_data = []

    prev_material_balance = None

    while not board.is_game_over(claim_draw=True):
        state = board_to_tensor(board)
        if board.turn == chess.WHITE:
            move = ai_white.select_move(board)
            player = 'white'
        else:
            move = ai_black.select_move(board)
            player = 'black'
        move_history.append(move.uci())

        # Personality reward: calculate bonuses for DestroyerAI's moves
        bonus = 0
        if player == "black":
            if board.is_capture(move):
                bonus += 0.5
            board.push(move)
            if board.is_check():
                bonus += 0.2
            # Sacrifice bonus: lost material but not checkmate
            # Calculate material balance (sum of piece values)
            material = sum(
                [piece_value(piece.piece_type) for piece in board.piece_map().values() if piece.color == chess.BLACK]
            )
            if prev_material_balance is not None and material < prev_material_balance:
                bonus += 0.2
            prev_material_balance = material
            training_data.append((state, move, player, bonus))
        else:
            board.push(move)
            training_data.append((state, move, player, 0))  # No bonus for BestAI

    result = board.result()
    if result == "1-0":
        reward_white, reward_black = 1, -1
    elif result == "0-1":
        reward_white, reward_black = -1, 1
    else:
        reward_white, reward_black = 0, 0

    labeled_data = []
    for state, move, player, bonus in training_data:
        if player == 'white':
            reward = reward_white
        else:
            reward = reward_black + bonus
        labeled_data.append((state, move, reward))

    return result, move_history, labeled_data

def piece_value(pt):
    # Assign simple values to pieces: pawn=1, knight/bishop=3, rook=5, queen=9, king=0
    if pt == chess.PAWN:
        return 1
    if pt == chess.KNIGHT or pt == chess.BISHOP:
        return 3
    if pt == chess.ROOK:
        return 5
    if pt == chess.QUEEN:
        return 9
    return 0