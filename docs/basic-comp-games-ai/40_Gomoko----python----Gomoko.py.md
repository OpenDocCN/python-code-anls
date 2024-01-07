# `basic-computer-games\40_Gomoko\python\Gomoko.py`

```

# 导入随机模块和类型提示模块
import random
from typing import Any, List, Tuple


# 打印游戏棋盘
def print_board(A: List[List[Any]], n: int) -> None:
    """PRINT THE BOARD"""
    for i in range(n):
        print(" ", end="")
        for j in range(n):
            print(A[i][j], end="")
            print(" ", end="")
        print()


# 检查移动是否合法
def check_move(_I, _J, _N) -> bool:  # 910
    if _I < 1 or _I > _N or _J < 1 or _J > _N:
        return False
    return True


# 打印游戏横幅
def print_banner() -> None:
    # 打印游戏横幅信息
    print(" " * 33 + "GOMOKU")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    print("WELCOME TO THE ORIENTAL GAME OF GOMOKO.\n")
    # 打印游戏规则
    print("THE GAME IS PLAYED ON AN N BY N GRID OF A SIZE")
    print("THAT YOU SPECIFY.  DURING YOUR PLAY, YOU MAY COVER ONE GRID")
    print("INTERSECTION WITH A MARKER. THE OBJECT OF THE GAME IS TO GET")
    print("5 ADJACENT MARKERS IN A ROW -- HORIZONTALLY, VERTICALLY, OR")
    print("DIAGONALLY.  ON THE BOARD DIAGRAM, YOUR MOVES ARE MARKED")
    print("WITH A '1' AND THE COMPUTER MOVES WITH A '2'.\n")
    print("THE COMPUTER DOES NOT KEEP TRACK OF WHO HAS WON.")
    print("TO END THE GAME, TYPE -1,-1 FOR YOUR MOVE.\n")


# 获取游戏棋盘尺寸
def get_board_dimensions() -> int:
    n = 0
    while True:
        n = int(input("WHAT IS YOUR BOARD SIZE (MIN 7/ MAX 19)? "))
        if n < 7 or n > 19:
            print("I SAID, THE MINIMUM IS 7, THE MAXIMUM IS 19.")
            print()
        else:
            break
    return n


# 获取玩家的移动
def get_move() -> Tuple[int, int]:
    while True:
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


# 初始化游戏棋盘
def initialize_board(n: int) -> List[List[int]]:
    # 初始化棋盘
    board = []
    for _x in range(n):
        sub_a = []
        for _y in range(n):
            sub_a.append(0)
        board.append(sub_a)
    return board


# 主函数
def main() -> None:
    # 打印游戏横幅
    print_banner()

    while True:
        # 获取游戏棋盘尺寸
        n = get_board_dimensions()
        # 初始化游戏棋盘
        board = initialize_board(n)

        print()
        print()
        print("WE ALTERNATE MOVES. YOU GO FIRST...")
        print()

        while True:
            # 获取玩家的移动
            x, y = get_move()
            if x == -1:
                break
            elif not check_move(x, y, n):
                print("ILLEGAL MOVE.  TRY AGAIN...")
            else:
                if board[x - 1][y - 1] != 0:
                    print("SQUARE OCCUPIED.  TRY AGAIN...")
                else:
                    board[x - 1][y - 1] = 1
                    # 计算计算机的智能移动
                    skip_ef_loop = False
                    for E in range(-1, 2):
                        for F in range(-1, 2):
                            if E + F - E * F == 0 or skip_ef_loop:
                                continue
                            X = x + F
                            Y = y + F
                            if not check_move(X, Y, n):
                                continue
                            if board[X - 1][Y - 1] == 1:
                                skip_ef_loop = True
                                X = x - E
                                Y = y - F
                                if not check_move(X, Y, n):  # 750
                                    while True:  # 610
                                        X = random.randint(1, n)
                                        Y = random.randint(1, n)
                                        if (
                                            check_move(X, Y, n)
                                            and board[X - 1][Y - 1] == 0
                                        ):
                                            board[X - 1][Y - 1] = 2
                                            print_board(board, n)
                                            break
                                else:
                                    if board[X - 1][Y - 1] != 0:
                                        while True:
                                            X = random.randint(1, n)
                                            Y = random.randint(1, n)
                                            if (
                                                check_move(X, Y, n)
                                                and board[X - 1][Y - 1] == 0
                                            ):
                                                board[X - 1][Y - 1] = 2
                                                print_board(board, n)
                                                break
                                    else:
                                        board[X - 1][Y - 1] = 2
                                        print_board(board, n)
        print()
        print("THANKS FOR THE GAME!!")
        repeat = int(input("PLAY AGAIN (1 FOR YES, 0 FOR NO)? "))
        if repeat == 0:
            break


if __name__ == "__main__":
    main()

```