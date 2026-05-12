from sudoku_generator import generate_puzzle
from agent import SudokuAgent
import torch
import torch.nn as nn
from collections import deque


def pretrain(model: SudokuAgent, opt, epochs: int, val_every: int = 200, savefile: str = "models/sudokuro_pretrain.pt"):
    loss_ma = deque(maxlen=20)
    val_puzzle, val_solution = generate_puzzle(72)

    for epoch in range(epochs):
        p, s = generate_puzzle(72)
        x = torch.tensor(p, dtype=torch.float32).unsqueeze(0) / 9.0
        y = (torch.tensor(s, dtype=torch.long).view(1, 81) - 1)
        mask = torch.tensor(p) == 0

        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits[0, mask], y[0, mask])

        with torch.no_grad():
            preds = logits[0, mask].argmax(dim=1)
            targets = y[0, mask]
            cell_acc = (preds == targets).float().mean().item()

            # conflict_rate = count_conflicts_in_prediction(p, preds, mask, s)

            loss_ma.append(loss.item())

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % 50 == 0:
            with torch.no_grad():
                preds = logits[0, mask].argmax(dim=1)
                acc = (preds == y[0, mask]).float().mean().item()
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.3f} | CellAcc: {acc:.2%}")

        if val_every > 0 and epoch % val_every == 0 and epoch > 0:
            val_metrics = validate(model, val_puzzle, val_solution)
            print(f"  └─ VAL | CellAcc: {val_metrics['acc']:.2%} | "
                  f"FullBoard: {val_metrics['solved']} | Conflicts: {val_metrics['conflicts']:.1%}")

    print(f"✅ Pretraining finished. Final CellAcc: {cell_acc:.2%}")
    torch.save(model.state_dict(), savefile)


def count_conflicts_in_prediction(puzzle, preds, mask, solution):
    board = puzzle.copy()
    filled_indices = mask.nonzero(as_tuple=True)[0].tolist()

    for idx, cell_idx in enumerate(filled_indices):
        board[cell_idx] = preds[idx].item() + 1

    return count_conflicts(board) / max(1, len(filled_indices))


def count_conflicts(board):
    conflicts = 0
    for i in range(81):
        if board[i] == 0: continue
        r, c = divmod(i, 9)
        br, bc = 3 * (r // 3), 3 * (c // 3)

        for k in range(9):
            if k != c and board[r * 9 + k] == board[i]:
                conflicts += 1
        for k in range(9):
            if k != r and board[k * 9 + c] == board[i]:
                conflicts += 1
        for rr in range(br, br + 3):
            for cc in range(bc, bc + 3):
                if (rr != r or cc != c) and board[rr * 9 + cc] == board[i]:
                    conflicts += 1
    return conflicts // 2


def validate(model: SudokuAgent, puzzle, solution):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(puzzle, dtype=torch.float32).unsqueeze(0) / 9.0
        mask = torch.tensor(puzzle) == 0
        logits = model(x).view(1, 81, 9)

        preds = logits[0, mask].argmax(dim=1)
        targets = torch.tensor(solution, dtype=torch.long)[mask]
        cell_acc = (preds == targets).float().mean().item()

        board = puzzle.copy()
        filled_idx = mask.nonzero(as_tuple=True)[0].tolist()
        for idx, cell in enumerate(filled_idx):
            board[cell] = preds[idx].item() + 1

        conflicts = count_conflicts(board) / max(1, len(filled_idx))
        solved = (board == solution)

    model.train()
    return {'acc': cell_acc, 'solved': solved, 'conflicts': conflicts}