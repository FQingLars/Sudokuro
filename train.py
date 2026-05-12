import torch
import torch.distributions as D
import random
from agent import SudokuAgent
from sudoku_generator import generate_puzzle
import collections


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

def train(model: SudokuAgent, opt, gamma: float = 0.99, start_baseline: float = 0.0, alpha_bl: float = 0.1, entropy_coef: float = 0.02, savename: str = "sudokuro.pth"):
    try:
        baseline = start_baseline

        clues = 75
        success_window = collections.deque(maxlen=100)

        last_clue_change = 0
        MIN_EP_BETWEEN_CHANGES = 200

        ep = 0
        while True:
            puzzle, solution = generate_puzzle(clues)
            board = puzzle.copy()
            log_probs, rewards = [], []

            while True:
                empties = [i for i, v in enumerate(board) if v == 0]
                if not empties: break

                cell = random.choice(empties)
                mask = get_valid_mask(board, cell)
                if not mask.any(): break

                state = torch.tensor(board, dtype=torch.float32).unsqueeze(0) / 9.0

                logits = model(state)
                cell_logits = logits[0, cell]
                masked_logits = cell_logits.clone()
                masked_logits[~mask] = -1e9

                dist = D.Categorical(logits=masked_logits)
                action = dist.sample()
                entropy = dist.entropy()

                log_probs.append(dist.log_prob(action))
                board[cell] = action.item() + 1
                step_reward = 1.0 if board[cell] == solution[cell] else -1.0
                rewards.append(step_reward)

            solved = (board == solution)
            if solved:
                rewards[-1] += 20.0
            else:
                rewards[-1] -= 10.0

            success_window.append(solved)
            success_rate = sum(success_window) / len(success_window)

            if ep - last_clue_change >= MIN_EP_BETWEEN_CHANGES:
                if success_rate > 0.75 and clues > 12:
                    clues -= 1
                    last_clue_change = ep
                    print(f"  └─ 📈 Difficulty ↑: {clues + 1} → {clues} clues")
                elif success_rate < 0.40 and clues < 80:
                    clues += 1
                    last_clue_change = ep
                    print(f"  └─ 📉 Difficulty ↓: {clues - 1} → {clues} clues")

            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            returns = torch.tensor(returns, dtype=torch.float32)
            episode_return = returns[0].item()
            baseline = (1 - alpha_bl) * baseline + alpha_bl * episode_return
            advantages = returns - baseline

            loss = -(torch.stack(log_probs) * advantages).sum() - entropy_coef * entropy.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if ep % 50 == 0:
                print(f"Ep {ep:5d} | Ret: {episode_return:6.1f} | WinRate: {success_rate:.2%} | Clues: {clues:2d} | {'✅' if solved else '❌'}")

            ep+=1
    except KeyboardInterrupt:
        print("Keyboard Interrupted.")
    finally:
        torch.save(model.state_dict(), savename)