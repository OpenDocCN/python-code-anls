# `basic-computer-games\48_High_IQ\python\High_IQ.py`

```

# 导入类型提示模块 Dict
from typing import Dict

# 创建新的棋盘，使用字典存储，键为整数，值为字符串
def new_board() -> Dict[int, str]:
    """
    Using a dictionary in python to store the board,
    since we are not including all numbers within a given range.
    """
    # 返回包含棋盘初始状态的字典
    return {
        # 棋盘上的位置和对应的棋子状态
        13: "!",
        14: "!",
        # ... 其他位置和状态
        41: "O",
    }

# 打印游戏说明
def print_instructions() -> None:
    # 打印游戏说明
    print(
        """
        ...
        """
    )

# 打印棋盘状态
def print_board(board: Dict[int, str]) -> None:
    """Prints the boards using indexes in the passed parameter"""
    # 根据传入的棋盘字典打印棋盘状态
    print(" " * 2 + board[13] + board[14] + board[15])
    # ... 其他行的打印

# 开始游戏
def play_game() -> None:
    # 创建新的棋盘
    board = new_board()

    # 主游戏循环
    while not is_game_finished(board):
        # 打印当前棋盘状态
        print_board(board)
        # 用户进行移动，直到合法移动为止
        while not move(board):
            print("ILLEGAL MOVE! TRY AGAIN")

    # 检查剩余棋子数量并打印用户得分
    peg_count = 0
    for key in board.keys():
        if board[key] == "!":
            peg_count += 1

    print("YOU HAD " + str(peg_count) + " PEGS REMAINING")

    if peg_count == 1:
        print("BRAVO! YOU MADE A PERFECT SCORE!")
        print("SAVE THIS PAPER AS A RECORD OF YOUR ACCOMPLISHMENT!")

# 用户移动棋子
def move(board: Dict[int, str]) -> bool:
    """Queries the user to move. Returns false if the user puts in an invalid input or move, returns true if the move was successful"""
    # 获取用户输入的起始位置
    start_input = input("MOVE WHICH PIECE? ")

    # 检查输入是否为数字
    if not start_input.isdigit():
        return False

    start = int(start_input)

    # 检查起始位置是否合法
    if start not in board or board[start] != "!":
        return False

    # 获取用户输入的目标位置
    end_input = input("TO WHERE? ")

    # 检查输入是否为数字
    if not end_input.isdigit():
        return False

    end = int(end_input)

    # 检查目标位置是否合法
    if end not in board or board[end] != "O":
        return False

    # 计算移动的距离和中间位置
    difference = abs(start - end)
    center = int((end + start) / 2)
    # 检查移动是否合法
    if (
        (difference == 2 or difference == 18)
        and board[end] == "O"
        and board[center] == "!"
    ):
        board[start] = "O"
        board[center] = "O"
        board[end] = "!"
        return True
    else:
        return False

# 主函数
def main() -> None:
    # 打印游戏标题
    print(" " * 33 + "H-I-Q")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    # 打印游戏说明
    print_instructions()
    # 开始游戏
    play_game()

# 检查游戏是否结束
def is_game_finished(board) -> bool:
    """Check all locations and whether or not a move is possible at that location."""
    # 遍历棋盘上的每个位置，检查是否还有合法的移动
    for pos in board.keys():
        if board[pos] == "!":
            for space in [1, 9]:
                # 检查相邻位置是否有棋子
                next_to_peg = ((pos + space) in board) and board[pos + space] == "!"
                # 检查是否有可移动的空位
                has_movable_space = (
                    not ((pos - space) in board and board[pos - space] == "!")
                ) or (
                    not ((pos + space * 2) in board and board[pos + space * 2] == "!")
                )
                if next_to_peg and has_movable_space:
                    return False
    return True

# 程序入口
if __name__ == "__main__":
    main()

```