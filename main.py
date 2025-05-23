# Where you start training and orchestrating games

# Entrypoint for training and orchestrating self-play

from self_play import self_play_game

if __name__ == "__main__":
    num_games = 10  # Number of self-play games to run
    for i in range(num_games):
        print(f"=== Game {i+1} ===")
        result, moves, labeled_data = self_play_game()
        print(f"Result: {result}")
        if result == "1-0":
            print("White (BestAI) wins!")
        elif result == "0-1":
            print("Black (DestroyerAI) wins!")
        else:
            print("It's a draw!")