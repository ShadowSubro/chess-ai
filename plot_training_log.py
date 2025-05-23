import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv("logs/training_log.csv")

# If you have multiple runs, filter or reset cycle numbers as needed
df = df[df['cycle'] <= 50]

plt.figure(figsize=(12,6))
plt.plot(df['cycle'], df['white_wins'], label='White Wins', marker='o')
plt.plot(df['cycle'], df['black_wins'], label='Black Wins', marker='o')
plt.plot(df['cycle'], df['draws'], label='Draws', marker='o')
plt.xlabel('Cycle')
plt.ylabel('Games')
plt.title('Chess AI Self-Play Training Progress')
plt.legend()
plt.grid()
plt.show()