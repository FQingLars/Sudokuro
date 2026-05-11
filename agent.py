import torch
import torch.nn as nn
import torch.distributions as D
import random
from sudoku_generator import generate_puzzle

class SudokuAgent(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(81, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, 9)
        )

    def forward(self, state):
        return self.net(state)

def get_valid_mask(board, cell_idx):
    row, col = divmod(cell_idx, 9)
    box_r, box_c = 3 * (row // 3), 3 * (col // 3)
    mask = torch.ones(9, dtype=torch.bool)

    for c in range(9):
        val = board[row * 9 + c]
        if val > 0: mask[val - 1] = False

    for r in range(9):
        val = board[r * 9 + col]
        if val > 0: mask[val - 1] = False

    for r in range(box_r, box_r + 3):
        for c in range(box_c, box_c + 3):
            val = board[r * 9 + c]
            if val > 0: mask[val - 1] = False

    return mask

def dynamic_difficulty(ep: int) -> int:
    if ep < 5000:
        return 60
    elif ep < 10000:
        return 50
    elif ep < 15000:
        return 40
    elif ep <= 30000:
        return 30
    else:
        return 15

agent = SudokuAgent(hidden=128)
opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
gamma = 0.99

baseline = 0.0
alpha_bl = 0.1

try:
    for ep in range(30000):
        clues = dynamic_difficulty(ep)
        puzzle, solution = generate_puzzle(clues)
        board = puzzle.copy()
        log_probs, rewards = [], []

        while True:
            empties = [i for i, v in enumerate(board) if v == 0]
            if not empties:
                break

            cell = random.choice(empties)
            mask = get_valid_mask(board, cell)
            if not mask.any():
                break

            state = torch.tensor(board, dtype=torch.float32).unsqueeze(0) / 9.0
            logits = agent(state).squeeze(0)

            masked_logits = logits.clone()
            masked_logits[~mask] = -1e9

            dist = D.Categorical(logits=masked_logits)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            board[cell] = action.item() + 1

            step_reward = 1.0 if board[cell] == solution[cell] else -1.0
            rewards.append(step_reward)

        solved = (board == solution)
        if solved:
            rewards[-1] += 20.0
        else:
            rewards[-1] -= 10.0

        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        baseline = (1 - alpha_bl) * baseline + alpha_bl * returns.mean().item()
        advantages = returns - baseline

        loss = -(torch.stack(log_probs) * advantages).sum()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()

        if ep % 100 == 0:
            print(f"Ep {ep:4d} | Ret: {returns.sum().item():6.1f} | Solved: {'✅' if solved else '❌'}")
except KeyboardInterrupt:
    print("Keyboard Interrupted.")
finally:
    torch.save(agent.state_dict(), "ultimate_sudoku.pth")