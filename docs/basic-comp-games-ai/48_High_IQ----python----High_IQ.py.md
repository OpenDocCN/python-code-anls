# `basic-computer-games\48_High_IQ\python\High_IQ.py`

```
# 导入类型提示模块 Dict
from typing import Dict

# 创建一个新的棋盘，使用字典来存储棋盘，因为我们不是在给定范围内包含所有数字
def new_board() -> Dict[int, str]:
    return {
        13: "!",
        14: "!",
        15: "!",
        22: "!",
        23: "!",
        24: "!",
        29: "!",
        30: "!",
        31: "!",
        32: "!",
        33: "!",
        34: "!",
        35: "!",
        38: "!",
        39: "!",
        40: "!",
        42: "!",
        43: "!",
        44: "!",
        47: "!",
        48: "!",
        49: "!",
        50: "!",
        51: "!",
        52: "!",
        53: "!",
        58: "!",
        59: "!",
        60: "!",
        67: "!",
        68: "!",
        69: "!",
        41: "O",
    }

# 打印游戏说明
def print_instructions() -> None:
    print(
        """
HERE IS THE BOARD:

          !    !    !
         13   14   15

          !    !    !
         22   23   24

!    !    !    !    !    !    !
29   30   31   32   33   34   35

!    !    !    !    !    !    !
38   39   40   41   42   43   44

!    !    !    !    !    !    !
47   48   49   50   51   52   53

          !    !    !
         58   59   60

          !    !    !
         67   68   69

TO SAVE TYPING TIME, A COMPRESSED VERSION OF THE GAME BOARD
WILL BE USED DURING PLAY.  REFER TO THE ABOVE ONE FOR PEG
NUMBERS.  OK, LET'S BEGIN.
    """
    )

# 打印棋盘，使用传入参数中的索引来打印棋盘
def print_board(board: Dict[int, str]) -> None:
    """Prints the boards using indexes in the passed parameter"""
    print(" " * 2 + board[13] + board[14] + board[15])
    print(" " * 2 + board[22] + board[23] + board[24])
    print(
        board[29]
        + board[30]
        + board[31]
        + board[32]
        + board[33]
        + board[34]
        + board[35]
    )
    print(
        board[38]
        + board[39]
        + board[40]
        + board[41]
        + board[42]
        + board[43]
        + board[44]
    )
    # 打印第一行的棋盘格局
    print(
        board[47]  # 打印第一行的第一个格子
        + board[48]  # 打印第一行的第二个格子
        + board[49]  # 打印第一行的第三个格子
        + board[50]  # 打印第一行的第四个格子
        + board[51]  # 打印第一行的第五个格子
        + board[52]  # 打印第一行的第六个格子
        + board[53]  # 打印第一行的第七个格子
    )
    # 打印第二行的棋盘格局
    print(" " * 2 + board[58] + board[59] + board[60])  # 打印第二行的三个格子
    # 打印第三行的棋盘格局
    print(" " * 2 + board[67] + board[68] + board[69])  # 打印第三行的三个格子
# 定义一个函数，用于玩游戏
def play_game() -> None:
    # 创建新的游戏板
    board = new_board()

    # 主游戏循环
    while not is_game_finished(board):
        # 打印游戏板
        print_board(board)
        # 当移动不合法时，提示用户重新输入
        while not move(board):
            print("ILLEGAL MOVE! TRY AGAIN")

    # 检查剩余的棋子数量并打印用户的得分
    peg_count = 0
    for key in board.keys():
        if board[key] == "!":
            peg_count += 1

    print("YOU HAD " + str(peg_count) + " PEGS REMAINING")

    # 如果剩余的棋子数量为1，则打印完美得分的消息
    if peg_count == 1:
        print("BRAVO! YOU MADE A PERFECT SCORE!")
        print("SAVE THIS PAPER AS A RECORD OF YOUR ACCOMPLISHMENT!")


# 定义一个函数，用于移动棋子
def move(board: Dict[int, str]) -> bool:
    """Queries the user to move. Returns false if the user puts in an invalid input or move, returns true if the move was successful"""
    # 提示用户输入起始位置
    start_input = input("MOVE WHICH PIECE? ")

    # 如果输入不是数字，则返回移动失败
    if not start_input.isdigit():
        return False

    start = int(start_input)

    # 如果起始位置不在游戏板上或者起始位置不是棋子，则返回移动失败
    if start not in board or board[start] != "!":
        return False

    # 提示用户输入目标位置
    end_input = input("TO WHERE? ")

    # 如果输入不是数字，则返回移动失败
    if not end_input.isdigit():
        return False

    end = int(end_input)

    # 如果目标位置不在游戏板上或者目标位置不是空位，则返回移动失败
    if end not in board or board[end] != "O":
        return False

    difference = abs(start - end)
    center = int((end + start) / 2)
    # 如果移动合法，则更新游戏板并返回移动成功
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


# 定义一个函数，用于主程序
def main() -> None:
    print(" " * 33 + "H-I-Q")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print_instructions()
    play_game()


# 定义一个函数，用于检查游戏是否结束
def is_game_finished(board) -> bool:
    """Check all locations and whether or not a move is possible at that location."""
    # 遍历棋盘上的每个位置
    for pos in board.keys():
        # 如果当前位置有一个棋子
        if board[pos] == "!":
            # 遍历可能的移动方向：向前和向后
            for space in [1, 9]:
                # 检查下一个位置是否有一个棋子
                next_to_peg = ((pos + space) in board) and board[pos + space] == "!"
                # 检查向前或向后移动的位置是否有可移动的空间
                has_movable_space = (
                    not ((pos - space) in board and board[pos - space] == "!")
                ) or (
                    not ((pos + space * 2) in board and board[pos + space * 2] == "!")
                )
                # 如果下一个位置有棋子并且有可移动的空间，则返回 False
                if next_to_peg and has_movable_space:
                    return False
    # 如果没有找到任何可移动的棋子，则返回 True
    return True
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```