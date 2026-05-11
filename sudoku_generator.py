import random

def _is_valid(board, r, c, n):
    for i in range(9):
        if board[r][i] == n or board[i][c] == n: return False
        if board[3*(r//3)+i//3][3*(c//3)+i%3] == n: return False
    return True

def _fill(board):
    for i in range(81):
        r, c = divmod(i, 9)
        if board[r][c] == 0:
            for n in random.sample(range(1, 10), 9):
                if _is_valid(board, r, c, n):
                    board[r][c] = n
                    if _fill(board): return True
                    board[r][c] = 0
            return False
    return True

def _count_solutions(board, idx=0):
    if idx == 81: return 1
    r, c = divmod(idx, 9)
    if board[r][c] != 0: return _count_solutions(board, idx+1)
    cnt = 0
    for n in range(1, 10):
        if _is_valid(board, r, c, n):
            board[r][c] = n
            cnt += _count_solutions(board, idx+1)
            if cnt > 1: return cnt
            board[r][c] = 0
    return cnt

def generate_puzzle(clues=30):
    board = [[0]*9 for _ in range(9)]
    _fill(board)
    solution = [n for row in board for n in row]

    cells = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(cells)

    for r, c in cells:
        if sum(1 for row in board for x in row if x != 0) <= clues: break
        val = board[r][c]
        board[r][c] = 0
        if _count_solutions([row[:] for row in board]) != 1:
            board[r][c] = val

    return [n for row in board for n in row], solution