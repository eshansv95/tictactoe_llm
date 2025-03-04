import torch
from torch.utils.data import Dataset
import json

class TicTacToeDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, "r") as f:
            self.data = json.load(f)  # ✅ Load data directly from JSON file

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = self.data[idx]["board"]
        move_probs = self.data[idx]["probabilities"]

        # ✅ Convert lists to tensors (ensure floats)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        move_probs_tensor = torch.tensor(move_probs, dtype=torch.float32)

        return state_tensor, move_probs_tensor




## always get the data set first decide evaluation based on the data set type dont do the bs i did 