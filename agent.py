import torch.nn as nn

class SudokuAgent(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(81, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),

            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),

            nn.Linear(hidden, 81 * 9)
        )

    def forward(self, state):
        x = self.net(state)
        return x.view(-1, 81, 9)