from pretrain import pretrain
from agent import SudokuAgent
from train import train
import torch.optim as optim
import torch
import os


MODEL = SudokuAgent(256)
OPT = optim.Adam(MODEL.parameters(), lr=0.001)
PRETRAIN_PATH = "models/sudokuro_pretrain.pt"


if __name__ == "__main__":
    #if not os.path.exists(PRETRAIN_PATH):
    #    pretrain(MODEL, OPT, 2000)
    #else:
    #    state = torch.load(PRETRAIN_PATH, weights_only=True)
    #    MODEL.load_state_dict(state)

    train(MODEL, OPT)
