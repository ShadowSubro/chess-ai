import os
from tqdm import trange
from self_play import self_play_game
from ai_best import BestAI
from ai_destroyer import DestroyerAI
import torch
import torch.optim as optim

def train_ai(ai_class, labeled_data, model_path, epochs=1, lr=1e-3):
    ai = ai_class(model_path=model_path)
    model = ai.model
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for state, move, reward in labeled_data:
            input_tensor = state.unsqueeze(0)
            _, pred_value = model(input_tensor)
            target = torch.tensor([[reward]], dtype=torch.float)
            loss = loss_fn(pred_value, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    ai.save_model(model_path)

if __name__ == "__main__":
    num_cycles = 150     # Number of train cycles (increase for more learning)
    games_per_cycle = 10    # Number of self-play games per cycle
    model_best = "BestAI_model.pth"
    model_destroyer = "DestroyerAI_model.pth"
    log_file = "logs/training_log.csv"

    # Write header for CSV log
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("cycle,white_wins,black_wins,draws\n")

    for cycle in range(num_cycles):
        print(f"\n=== Training Cycle {cycle+1} ===")
        all_white_data, all_black_data = [], []
        white_wins = black_wins = draws = 0

        # Self-play batch of games
        for i in trange(games_per_cycle, desc="Self-play games"):
            result, moves, labeled_data = self_play_game()
            if result == "1-0":
                white_wins += 1
            elif result == "0-1":
                black_wins += 1
            else:
                draws += 1
            all_white_data.extend([x for x in labeled_data if x[2] > 0 or x[2] == 0])
            all_black_data.extend([x for x in labeled_data if x[2] < 0 or x[2] == 0])

        # Log the results of this cycle
        with open(log_file, "a") as f:
            f.write(f"{cycle+1},{white_wins},{black_wins},{draws}\n")

        print(f"Cycle {cycle+1} results: White Wins: {white_wins}, Black Wins: {black_wins}, Draws: {draws}")
        print("Training BestAI (White)...")
        train_ai(BestAI, all_white_data, model_best, epochs=2)
        print("Training DestroyerAI (Black)...")
        train_ai(DestroyerAI, all_black_data, model_destroyer, epochs=2)
        print(f"Finished cycle {cycle+1}, models saved.\n")