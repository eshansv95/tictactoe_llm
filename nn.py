import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TicTacToeNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 9)

    def forward(self, x):
        x = x.float()  # ✅ Ensure input is float
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)  # ✅ Needed for KLDivLoss
        return x

def initialize_model():
    model = TicTacToeNN()
    criterion = nn.KLDivLoss(reduction="batchmean")  # ✅ KLDivLoss for multiple good moves
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    return model, criterion, optimizer

if __name__ == "__main__":
    model, criterion, optimizer = initialize_model()
    print("✅ Neural network, loss function & optimizer are set up correctly!")


# cross entropy is only good for a single move
# kl loss divergence is better cus there might be multiple good moves 
# we give the output as a vector table of 9 probabilities
