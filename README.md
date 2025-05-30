# Agar.io Clone with Q-Learning Bots

This is a simple clone of the popular game Agar.io, featuring Q-learning bots that learn to play the game using PyTorch.

## Features

- Player-controlled blob that can move and split
- AI-controlled bots that learn using Q-learning
- Collectible food particles
- Collision detection between players and bots
- Dynamic size increase when consuming food or other players

## Requirements

- Python 3.8+
- PyGame
- PyTorch
- NumPy

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## How to Play

Run the game:
```bash
python game.py
```

### Controls
- Arrow keys: Move your blob
- Space: Split your blob (when large enough)

### Game Rules
- Collect food particles to grow larger
- Larger blobs can eat smaller ones
- Avoid larger blobs that can eat you
- Watch the AI bots learn and adapt their strategies

## Q-Learning Implementation

The bots use Deep Q-Learning with the following features:
- State space includes distances to nearest food, walls, and other players
- Action space consists of four possible movements (up, down, left, right)
- Reward function based on:
  - Proximity to food (positive reward)
  - Proximity to larger players (negative reward)
  - Proximity to smaller players (positive reward)
- Experience replay for better learning
- Target network for stability

## Contributing

Feel free to submit issues and enhancement requests! 