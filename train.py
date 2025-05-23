from self_play import self_play_game
from ai_best import BestAI
from ai_destroyer import DestroyerAI
import torch
import torch.optim as optim

def train_ai(ai_class, labeled_data, epochs=1, lr=1e-3):
    ai = ai_class()
    model = ai.model
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        for state, move, reward in labeled_data:
            # Forward pass: predict value
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
    result, moves, labeled_data = self_play_game()
    print("Training BestAI...")
    # Train only on moves by 'white' for BestAI, 'black' for DestroyerAI
    white_data = [item for item in labeled_data if item[2] == 1 or item[2] == 0]
    black_data = [item for item in labeled_data if item[2] == -1 or item[2] == 0]
    train_ai(BestAI, white_data)
    print("Training DestroyerAI...")
    train_ai(DestroyerAI, black_data)