# 🧠 Tic-Tac-Toe AI  

A **Neural Network-based Tic-Tac-Toe AI** that learns optimal moves from training data. The model aims to play the best possible game, leading to a **draw with perfect play**.  

---

## 📂 Project Structure  

- **`dataset.py`** → Loads and processes the training data (`training_data.json`).  
- **`nn.py`** → Defines the neural network architecture for Tic-Tac-Toe.  
- **`play.py`** → Lets you play against the trained AI.  
- **`training.py`** → Trains the model using the dataset.  
- **`tictactoe_ai.pth`** → The saved trained model.  
- **`training_data.json`** → Dataset containing board states and optimal moves.  

---

## 🚀 How to Use  

### 1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/yourusername/tic-tac-toe-ai.git
cd tic-tac-toe-ai
2️⃣ Install Dependencies

pip install torch numpy
3️⃣ Train the AI

python training.py
(This trains the model using training_data.json and saves it as tictactoe_ai.pth.)

4️⃣ Play Against the AI

python play.py
📌 Notes
Ensure you have Python 3.x installed.
You can modify training_data.json to fine-tune the AI's learning
