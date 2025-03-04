import torch
import torch.nn as nn
import torch.optim as optim
from nn import initialize_model  # ✅ Import model setup from nn.py
from dataset import TicTacToeDataset  # ✅ Import dataset class
from torch.utils.data import DataLoader
import os



# ✅ Load Model, Loss Function & Optimizer
model, criterion, optimizer = initialize_model()

if os.path.exists("tictactoe_ai.pth"):
    model.load_state_dict(torch.load("tictactoe_ai.pth"))
    print("✅ Loaded pre-trained model!")
else:
    print("🚀 Starting training from scratch!")

# ✅ Load Dataset
dataset = TicTacToeDataset("training_data.json")  # Ensure this file exists
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ✅ Training Loop
num_epochs = 100  # Number of times the model will see the full dataset

for epoch in range(num_epochs):
    total_loss = 0.0  # Track loss for this epoch

    for states, target_probs in dataloader:
        # ✅ Forward Pass
        outputs = model(states)  # Get model predictions (log probabilities)
        
        # ✅ Compute Loss (KLDivLoss expects log probabilities)
        loss = criterion(outputs, target_probs)
        
        # ✅ Backpropagation
        optimizer.zero_grad()  # Reset gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters
        
        total_loss += loss.item()  # Accumulate loss
    
    # ✅ Print Epoch Summary
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

print("🎉 Training Complete! Model is ready.")
torch.save(model.state_dict(), "tictactoe_ai.pth")  # ✅ Save the trained model
