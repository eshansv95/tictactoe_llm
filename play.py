import torch
import torch.nn.functional as F
from nn import TicTacToeNN  # Import your trained model

# ✅ Load trained AI model
model = TicTacToeNN()
model.load_state_dict(torch.load("tictactoe_ai.pth"))
model.eval()  # Set model to evaluation mode

def print_board(board):
    """Prints the Tic-Tac-Toe board in a readable format."""
    symbols = {1: "X", -1: "O", 0: " "}
    print("\nBoard:")
    for i in range(3):
        print(" | ".join(symbols[board[i * 3 + j]] for j in range(3)))
        if i < 2:
            print("-" * 9)

def get_ai_move(board):
    """AI predicts the best move based on trained model."""
    board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
    with torch.no_grad():
        move_probs = model(board_tensor).exp()  # Convert log probabilities to normal probabilities
    move_probs = move_probs.squeeze().numpy()

    # Choose the best available move
    best_move = max(
        (i for i in range(9) if board[i] == 0), 
        key=lambda i: move_probs[i], 
        default=None
    )
    return best_move

# ✅ Play against AI
board = [0] * 9  # Empty board
player = 1  # You are "X" (1), AI is "O" (-1)

while True:
    print_board(board)
    
    if player == 1:
        move = int(input("Enter your move (0-8): "))  # Get user input
        if board[move] != 0:
            print("Invalid move! Try again.")
            continue
    else:
        move = get_ai_move(board)  # AI move
        print(f"AI chooses: {move}")
    
    board[move] = player  # Update board
    player *= -1  # Switch turns
