import torch
import torch.optim as optim
from tqdm import trange
from self_play import self_play_game
from ai_best import BestAI
from ai_destroyer import DestroyerAI

def train_ai(ai_class, labeled_data, epochs=1, lr=1e-3):
    ai = ai_class()
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
    # Save updated model
    model_path = f"{ai_class.__name__}_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved: {model_path}")

if __name__ == "__main__":
    num_games = 20  # Try 20+ for better learning!
    all_white_data = []
    all_black_data = []

    for i in trange(num_games, desc="Playing games"):
        result, moves, labeled_data = self_play_game()
        # Separate data for each AI
        all_white_data.extend([x for x in labeled_data if x[2] == 1 or x[2] == 0])  # reward > 0 means white
        all_black_data.extend([x for x in labeled_data if x[2] == -1 or x[2] == 0]) # reward < 0 means black

    print("Training BestAI...")
    train_ai(BestAI, all_white_data, epochs=2)
    print("Training DestroyerAI...")
    train_ai(DestroyerAI, all_black_data, epochs=2)