# `d:/src/tocomm/basic-computer-games\40_Gomoko\python\Gomoko.py`

```
import random  # 导入 random 模块
from typing import Any, List, Tuple  # 从 typing 模块中导入 Any、List、Tuple 类型


def print_board(A: List[List[Any]], n: int) -> None:
    """PRINT THE BOARD"""
    # 打印游戏板
    for i in range(n):
        print(" ", end="")
        for j in range(n):
            print(A[i][j], end="")
            print(" ", end="")
        print()


def check_move(_I, _J, _N) -> bool:  # 910
    # 检查移动是否有效
    if _I < 1 or _I > _N or _J < 1 or _J > _N:
        return False
    return True
def print_banner() -> None:
    # 打印游戏横幅
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

def get_board_dimensions() -> int:
    n = 0
    while True:
        # 获取用户输入的游戏板大小
        n = int(input("WHAT IS YOUR BOARD SIZE (MIN 7/ MAX 19)? "))
        # 检查用户输入是否在规定范围内
        if n < 7 or n > 19:
            print("I SAID, THE MINIMUM IS 7, THE MAXIMUM IS 19.")
            print()
```
这行代码是一个空的print语句，它的作用是在控制台输出一个空行。

```
        else:
            break
```
这行代码是一个else语句，它的作用是在循环中如果条件不满足就跳出循环。

```
    return n
```
这行代码是一个return语句，它的作用是返回变量n的值。

```
def get_move() -> Tuple[int, int]:
```
这行代码是一个函数定义语句，定义了一个名为get_move的函数，它接受没有参数并且返回一个元组类型的值。

```
    while True:
```
这行代码是一个while循环语句，它的作用是创建一个无限循环，直到循环条件不满足时才会跳出循环。

```
        xy = input("YOUR PLAY (I,J)? ")
```
这行代码是一个input语句，它的作用是在控制台接收用户输入，并将输入的值赋给变量xy。

```
        print()
```
这行代码是一个空的print语句，它的作用是在控制台输出一个空行。

```
        x_str, y_str = xy.split(",")
```
这行代码是一个split语句，它的作用是将变量xy中的字符串按照逗号分隔，并将分隔后的两个部分分别赋给变量x_str和y_str。

```
        try:
            x = int(x_str)
            y = int(y_str)
        except Exception:
            print("ILLEGAL MOVE.  TRY AGAIN...")
            continue
```
这段代码是一个try-except语句，它的作用是尝试将x_str和y_str转换为整数类型，如果转换失败则捕获异常并输出错误信息，然后继续循环。

```
        return x, y
```
这行代码是一个return语句，它的作用是返回变量x和y的值作为元组类型的值。
def initialize_board(n: int) -> List[List[int]]:
    # 初始化棋盘
    board = []  # 创建一个空列表，用于存储棋盘
    for _x in range(n):  # 循环n次，创建n行
        sub_a = []  # 创建一个空列表，用于存储每一行的数据
        for _y in range(n):  # 循环n次，创建n列
            sub_a.append(0)  # 将0添加到每一行中，表示初始状态为0
        board.append(sub_a)  # 将每一行添加到棋盘中
    return board  # 返回初始化后的棋盘


def main() -> None:
    print_banner()  # 调用打印横幅的函数

    while True:  # 无限循环
        n = get_board_dimensions()  # 获取棋盘的尺寸
        board = initialize_board(n)  # 初始化棋盘

        print()  # 打印空行
        print()  # 再次打印空行
        # 打印提示信息，要求对手先行
        print("WE ALTERNATE MOVES. YOU GO FIRST...")
        # 打印空行
        print()

        # 进入游戏循环
        while True:
            # 获取玩家的移动坐标
            x, y = get_move()
            # 如果玩家输入 -1，则退出游戏
            if x == -1:
                break
            # 如果玩家输入的坐标不合法，提示重新输入
            elif not check_move(x, y, n):
                print("ILLEGAL MOVE.  TRY AGAIN...")
            else:
                # 如果玩家选择的位置已经被占据，提示重新输入
                if board[x - 1][y - 1] != 0:
                    print("SQUARE OCCUPIED.  TRY AGAIN...")
                else:
                    # 将玩家的选择标记在棋盘上
                    board[x - 1][y - 1] = 1
                    # 计算机尝试智能移动
                    skip_ef_loop = False
                    # 遍历周围的位置，寻找合适的智能移动
                    for E in range(-1, 2):
                        for F in range(-1, 2):
                            # 如果条件满足或者已经跳过了循环，则继续下一次循环
                            if E + F - E * F == 0 or skip_ef_loop:
                                continue
# 计算新的 X 坐标
X = x + F
# 计算新的 Y 坐标
Y = y + F
# 如果新的坐标不符合移动规则，则跳过本次循环
if not check_move(X, Y, n):
    continue
# 如果新的坐标上的值为 1，则设置标志位为 True，并计算新的坐标
if board[X - 1][Y - 1] == 1:
    skip_ef_loop = True
    X = x - E
    Y = y - F
    # 如果新的坐标不符合移动规则
    if not check_move(X, Y, n):  # 750
        # 进入循环，直到找到合适的位置
        while True:  # 610
            X = random.randint(1, n)
            Y = random.randint(1, n)
            # 如果找到合适的位置，则设置对应的值为 2，打印当前棋盘状态，然后跳出循环
            if (
                check_move(X, Y, n)
                and board[X - 1][Y - 1] == 0
            ):
                board[X - 1][Y - 1] = 2
                print_board(board, n)
                break
    else:
# 如果左上角的方块不为0，则进入循环
if board[X - 1][Y - 1] != 0:
    # 在X和Y的范围内生成随机数
    while True:
        X = random.randint(1, n)
        Y = random.randint(1, n)
        # 检查移动是否有效并且目标方块为空
        if (
            check_move(X, Y, n)
            and board[X - 1][Y - 1] == 0
        ):
            # 将目标方块设为2，并打印游戏板
            board[X - 1][Y - 1] = 2
            print_board(board, n)
            # 退出循环
            break
# 如果左上角的方块为0，则将目标方块设为2，并打印游戏板
else:
    board[X - 1][Y - 1] = 2
    print_board(board, n)

# 打印空行和结束语句
print()
print("THANKS FOR THE GAME!!")

# 询问是否再玩一次，如果输入0则退出循环
repeat = int(input("PLAY AGAIN (1 FOR YES, 0 FOR NO)? "))
if repeat == 0:
    break
# 如果当前脚本被直接执行而不是被导入，则执行main函数
if __name__ == "__main__":
    main()
```