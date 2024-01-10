# `basic-computer-games\40_Gomoko\python\Gomoko.py`

```
# 导入 random 模块
import random
# 导入 typing 模块中的 Any、List、Tuple 类型
from typing import Any, List, Tuple

# 定义函数，打印游戏棋盘
def print_board(A: List[List[Any]], n: int) -> None:
    """PRINT THE BOARD"""
    # 遍历棋盘，打印每个格子的内容
    for i in range(n):
        print(" ", end="")
        for j in range(n):
            print(A[i][j], end="")
            print(" ", end="")
        print()

# 定义函数，检查移动是否合法
def check_move(_I, _J, _N) -> bool:  # 910
    # 如果移动超出棋盘范围，则返回 False
    if _I < 1 or _I > _N or _J < 1 or _J > _N:
        return False
    # 否则返回 True
    return True

# 定义函数，打印游戏横幅
def print_banner() -> None:
    # 打印游戏标题
    print(" " * 33 + "GOMOKU")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    print("WELCOME TO THE ORIENTAL GAME OF GOMOKO.\n")
    print("THE GAME IS PLAYED ON AN N BY N GRID OF A SIZE")
    print("THAT YOU SPECIFY.  DURING YOUR PLAY, YOU MAY COVER ONE GRID")
    print("INTERSECTION WITH A MARKER. THE OBJECT OF THE GAME IS TO GET")
    print("5 ADJACENT MARKERS IN A ROW -- HORIZONTALLY, VERTICALLY, OR")
    print("DIAGONALLY.  ON THE BOARD DIAGRAM, YOUR MOVES ARE MARKED")
    print("WITH A '1' AND THE COMPUTER MOVES WITH A '2'.\n")
    print("THE COMPUTER DOES NOT KEEP TRACK OF WHO HAS WON.")
    print("TO END THE GAME, TYPE -1,-1 FOR YOUR MOVE.\n")

# 定义函数，获取棋盘尺寸
def get_board_dimensions() -> int:
    n = 0
    while True:
        # 获取用户输入的棋盘尺寸
        n = int(input("WHAT IS YOUR BOARD SIZE (MIN 7/ MAX 19)? "))
        # 如果尺寸不在规定范围内，则提示用户重新输入
        if n < 7 or n > 19:
            print("I SAID, THE MINIMUM IS 7, THE MAXIMUM IS 19.")
            print()
        else:
            break
    return n

# 定义函数，获取玩家的移动
def get_move() -> Tuple[int, int]:
    while True:
        # 获取玩家输入的移动坐标
        xy = input("YOUR PLAY (I,J)? ")
        print()
        x_str, y_str = xy.split(",")
        try:
            x = int(x_str)
            y = int(y_str)
        except Exception:
            print("ILLEGAL MOVE.  TRY AGAIN...")
            continue
        return x, y

# 定义函数，初始化棋盘
def initialize_board(n: int) -> List[List[int]]:
    # 初始化棋盘，将每个格子的值初始化为 0
    board = []
    for _x in range(n):
        sub_a = []
        for _y in range(n):
            sub_a.append(0)
        board.append(sub_a)
    return board
# 定义主函数，没有参数，没有返回值
def main() -> None:
    # 调用打印横幅的函数
    print_banner()

# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```