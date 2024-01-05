# `88_3-D_Tic-Tac-Toe\python\qubit.py`

```
#!/usr/bin/env python3  # 指定脚本的解释器为 Python 3

# Ported from the BASIC source for 3D Tic Tac Toe
# in BASIC Computer Games, by David H. Ahl
# The code originated from Dartmouth College

from enum import Enum  # 导入 Enum 类
from typing import Optional, Tuple, Union  # 导入类型提示模块

# 定义 Move 枚举类，表示游戏状态和机器移动的类型
class Move(Enum):
    """Game status and types of machine move"""

    HUMAN_WIN = 0  # 人类获胜
    MACHINE_WIN = 1  # 机器获胜
    DRAW = 2  # 平局
    MOVES = 3  # 移动
    LIKES = 4  # 喜欢
    TAKES = 5  # 拿走
    GET_OUT = 6  # 离开
    YOU_FOX = 7  # 定义常量 YOU_FOX 为 7
    NICE_TRY = 8  # 定义常量 NICE_TRY 为 8
    CONCEDES = 9  # 定义常量 CONCEDES 为 9


class Player(Enum):
    EMPTY = 0  # 定义枚举类型 Player，其中 EMPTY 的值为 0
    HUMAN = 1  # 定义枚举类型 Player，其中 HUMAN 的值为 1
    MACHINE = 2  # 定义枚举类型 Player，其中 MACHINE 的值为 2


class TicTacToe3D:
    """The game logic for 3D Tic Tac Toe and the machine opponent"""

    def __init__(self) -> None:
        # 4x4x4 board keeps track of which player occupies each place
        # and used by machine to work out its strategy
        self.board = [0] * 64  # 初始化一个长度为 64 的列表，用于记录每个位置上的玩家情况

        # starting move
        # 定义棋盘的角落位置
        self.corners = [0, 48, 51, 3, 12, 60, 63, 15, 21, 38, 22, 37, 25, 41, 26, 42]

        # 定义用于检查游戏结束的线
        self.lines = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23],
            [24, 25, 26, 27],
            [28, 29, 30, 31],
            [32, 33, 34, 35],
            [36, 37, 38, 39],
            [40, 41, 42, 43],
            [44, 45, 46, 47],
            [48, 49, 50, 51],
            [52, 53, 54, 55],
            [56, 57, 58, 59],
            [60, 61, 62, 63],
# 创建一个二维列表，包含一系列数字组合
num_list = [
    [0, 16, 32, 48],
    [4, 20, 36, 52],
    [8, 24, 40, 56],
    [12, 28, 44, 60],
    [1, 17, 33, 49],
    [5, 21, 37, 53],
    [9, 25, 41, 57],
    [13, 29, 45, 61],
    [2, 18, 34, 50],
    [6, 22, 38, 54],
    [10, 26, 42, 58],
    [14, 30, 46, 62],
    [3, 19, 35, 51],
    [7, 23, 39, 55],
    [11, 27, 43, 59],
    [15, 31, 47, 63],
    [0, 4, 8, 12],
    [16, 20, 24, 28],
    [32, 36, 40, 44],
    [48, 52, 56, 60],
]
# 创建一个二维列表，包含多个子列表，每个子列表包含四个整数
matrix = [
    [1, 5, 9, 13],
    [17, 21, 25, 29],
    [33, 37, 41, 45],
    [49, 53, 57, 61],
    [2, 6, 10, 14],
    [18, 22, 26, 30],
    [34, 38, 42, 46],
    [50, 54, 58, 62],
    [3, 7, 11, 15],
    [19, 23, 27, 31],
    [35, 39, 43, 47],
    [51, 55, 59, 63],
    [0, 5, 10, 15],
    [16, 21, 26, 31],
    [32, 37, 42, 47],
    [48, 53, 58, 63],
    [12, 9, 6, 3],
    [28, 25, 22, 19],
    [44, 41, 38, 35],
    [60, 57, 54, 51],
]
# 创建一个二维列表，包含20个子列表，每个子列表包含4个元素
matrix = [
    [0, 20, 40, 60],
    [1, 21, 41, 61],
    [2, 22, 42, 62],
    [3, 23, 43, 63],
    [48, 36, 24, 12],
    [49, 37, 25, 13],
    [50, 38, 26, 14],
    [51, 39, 27, 15],
    [0, 17, 34, 51],
    [4, 21, 38, 55],
    [8, 25, 42, 59],
    [12, 29, 46, 63],
    [48, 33, 18, 3],
    [52, 37, 22, 7],
    [56, 41, 26, 11],
    [60, 45, 30, 15],
    [0, 21, 42, 63],
    [15, 26, 37, 48],
    [3, 22, 41, 60],
    [12, 25, 38, 51]
]
        ]

    def get(self, x, y, z) -> Player:
        # 获取指定位置的值
        m = self.board[4 * (4 * z + y) + x]
        # 如果值为40，返回机器玩家
        if m == 40:
            return Player.MACHINE
        # 如果值为8，返回人类玩家
        elif m == 8:
            return Player.HUMAN
        # 否则返回空
        else:
            return Player.EMPTY

    def move_3d(self, x, y, z, player) -> bool:
        # 计算在3D棋盘中的位置
        m = 4 * (4 * z + y) + x
        # 调用move函数执行移动
        return self.move(m, player)

    def move(self, m, player) -> bool:
        # 如果指定位置的值大于1，返回False
        if self.board[m] > 1:
            return False

        # 如果玩家是机器玩家
        if player == Player.MACHINE:
            self.board[m] = 40  # 如果条件成立，将self.board中索引为m的位置设置为40
        else:
            self.board[m] = 8  # 如果条件不成立，将self.board中索引为m的位置设置为8
        return True  # 返回True

    def get_3d_position(self, m) -> Tuple[int, int, int]:
        x = m % 4  # 计算m在3D空间中的x坐标
        y = (m // 4) % 4  # 计算m在3D空间中的y坐标
        z = m // 16  # 计算m在3D空间中的z坐标
        return x, y, z  # 返回3D坐标

    def evaluate_lines(self) -> None:
        self.lineValues = [0] * 76  # 初始化self.lineValues列表为76个0
        for j in range(76):  # 遍历76次
            value = 0  # 初始化value为0
            for k in range(4):  # 遍历4次
                value += self.board[self.lines[j][k]]  # 将self.board中指定位置的值加到value上
            self.lineValues[j] = value  # 将计算得到的value存入self.lineValues中对应的位置

    def strategy_mark_line(self, i) -> None:
        for j in range(4):  # 循环4次，j的取值范围为0到3
            m = self.lines[i][j]  # 获取self.lines[i][j]的值，赋给m
            if self.board[m] == 0:  # 如果self.board[m]的值为0
                self.board[m] = 1  # 将self.board[m]的值设为1

    def clear_strategy_marks(self) -> None:
        for i in range(64):  # 循环64次，i的取值范围为0到63
            if self.board[i] == 1:  # 如果self.board[i]的值为1
                self.board[i] = 0  # 将self.board[i]的值设为0

    def mark_and_move(self, vlow, vhigh, vmove) -> Optional[Tuple[Move, int]]:
        """
        mark lines that can potentially win the game for the human
        or the machine and choose best place to play
        """
        for i in range(76):  # 循环76次，i的取值范围为0到75
            value = 0  # 初始化value为0
            for j in range(4):  # 循环4次，j的取值范围为0到3
                value += self.board[self.lines[i][j]]  # 将self.board[self.lines[i][j]]的值加到value上
            self.lineValues[i] = value  # 将value赋给self.lineValues[i]
        if vlow <= value < vhigh:  # 检查value是否在vlow和vhigh之间
            if value > vlow:  # 如果value大于vlow
                return self.move_triple(i)  # 调用move_triple方法并返回结果
            self.strategy_mark_line(i)  # 调用strategy_mark_line方法
        self.evaluate_lines()  # 调用evaluate_lines方法

        for i in range(76):  # 循环76次
            value = self.lineValues[i]  # 获取self.lineValues中第i个元素的值
            if value == 4 or value == vmove:  # 如果value等于4或者等于vmove
                return self.move_diagonals(i, 1)  # 调用move_diagonals方法并返回结果
        return None  # 返回None

    def machine_move(self) -> Union[None, Tuple[Move, int], Tuple[Move, int, int]]:
        """machine works out what move to play"""
        self.clear_strategy_marks()  # 调用clear_strategy_marks方法

        self.evaluate_lines()  # 调用evaluate_lines方法
        for value, event in [  # 遍历列表
            (32, self.human_win),  # (32, self.human_win)
            (120, self.machine_win),  # (120, self.machine_win)
        (24, self.block_human_win),  # 在策略列表中添加一个元组，元组包含一个整数和一个函数引用
    ]:
        for i in range(76):  # 遍历范围为0到75的整数
            if self.lineValues[i] == value:  # 如果self.lineValues列表中的第i个元素等于value
                return event(i)  # 返回一个事件

    m = self.mark_and_move(80, 88, 43)  # 调用mark_and_move函数，传入三个参数
    if m is not None:  # 如果m不是None
        return m  # 返回m

    self.clear_strategy_marks()  # 调用clear_strategy_marks函数

    m = self.mark_and_move(16, 24, 11)  # 调用mark_and_move函数，传入三个参数
    if m is not None:  # 如果m不是None
        return m  # 返回m

    for k in range(18):  # 遍历范围为0到17的整数
        value = 0  # 将value设置为0
        for i in range(4 * k, 4 * k + 4):  # 遍历范围为4*k到4*k+3的整数
            for j in range(4):  # 遍历范围为0到3的整数
                    value += self.board[self.lines[i][j]]
```这行代码将self.lines[i][j]对应的值加到value上。

```            if (32 <= value < 40) or (72 <= value < 80):```如果value的值在32到39之间或者72到79之间，则执行下面的代码。

```                for s in [1, 0]:```遍历列表[1, 0]中的元素。

```                    for i in range(4 * k, 4 * k + 4):```遍历范围从4*k到4*k+4的值。

```                        m = self.move_diagonals(i, s)```调用self.move_diagonals方法，传入i和s作为参数，将返回值赋给m。

```                        if m is not None:```如果m不是None，则执行下面的代码。

```                            return m```返回m的值。

```        self.clear_strategy_marks()```调用self.clear_strategy_marks方法。

```        for y in self.corners:```遍历self.corners中的元素。

```            if self.board[y] == 0:```如果self.board[y]的值为0，则执行下面的代码。

```                return (Move.MOVES, y)```返回(Move.MOVES, y)。

```        for i in range(64):```遍历范围从0到63的值。

```            if self.board[i] == 0:```如果self.board[i]的值为0，则执行下面的代码。

```                return (Move.LIKES, i)```返回(Move.LIKES, i)。

```        return (Move.DRAW, -1)```返回(Move.DRAW, -1)。
    def human_win(self, i) -> Tuple[Move, int, int]:
        # 返回一个元组，表示人类获胜的情况，包括Move类型、-1和i
        return (Move.HUMAN_WIN, -1, i)

    def machine_win(self, i) -> Optional[Tuple[Move, int, int]]:
        # 判断机器是否获胜，如果是则返回Move类型、m和i的元组，否则返回None
        for j in range(4):
            m = self.lines[i][j]
            if self.board[m] == 0:
                return (Move.MACHINE_WIN, m, i)
        return None

    def block_human_win(self, i) -> Optional[Tuple[Move, int]]:
        # 阻止人类获胜，如果可以阻止则返回Move类型和m的元组，否则返回None
        for j in range(4):
            m = self.lines[i][j]
            if self.board[m] == 0:
                return (Move.NICE_TRY, m)
        return None

    def move_triple(self, i) -> Tuple[Move, int]:
        """make two lines-of-3 or prevent human from doing this"""
        # 创建两条3行或者阻止人类这样做
        for j in range(4):
            m = self.lines[i][j]  # 从self.lines中获取特定位置的值
            if self.board[m] == 1:  # 检查self.board中对应位置的值是否为1
                if self.lineValues[i] < 40:  # 检查self.lineValues中特定位置的值是否小于40
                    return (Move.YOU_FOX, m)  # 如果满足条件，返回Move.YOU_FOX和m的元组
                else:
                    return (Move.GET_OUT, m)  # 如果不满足条件，返回Move.GET_OUT和m的元组
        return (Move.CONCEDES, -1)  # 如果以上条件都不满足，返回Move.CONCEDES和-1的元组

    # 在4x4方格的角落或中心盒中选择移动
    def move_diagonals(self, i, s) -> Optional[Tuple[Move, int]]:  # 定义函数move_diagonals，接受参数i和s，并返回一个元组或None
        if 0 < (i % 4) < 3:  # 检查i除以4的余数是否在1和2之间
            jrange = [1, 2]  # 如果满足条件，设置jrange为[1, 2]
        else:
            jrange = [0, 3]  # 如果不满足条件，设置jrange为[0, 3]
        for j in jrange:  # 遍历jrange中的值
            m = self.lines[i][j]  # 从self.lines中获取特定位置的值
            if self.board[m] == s:  # 检查self.board中对应位置的值是否等于s
                return (Move.TAKES, m)  # 如果满足条件，返回Move.TAKES和m的元组
        return None  # 如果以上条件都不满足，返回None
class Qubit:
    # 定义移动代码的方法，根据给定的棋盘和移动返回一个字符串
    def move_code(self, board, m) -> str:
        # 获取移动的三维位置
        x, y, z = board.get_3d_position(m)
        # 返回格式化后的字符串，表示移动的位置
        return f"{z + 1:d}{y + 1:d}{x + 1:d}"

    # 定义展示获胜的方法，根据给定的棋盘和索引展示获胜的移动
    def show_win(self, board, i) -> None:
        # 遍历指定索引的线上的移动
        for m in board.lines[i]:
            # 打印移动的代码
            print(self.move_code(board, m))

    # 定义展示棋盘的方法，根据给定的棋盘展示整个棋盘的状态
    def show_board(self, board) -> None:
        # 定义棋子的颜色
        c = " YM"
        # 遍历棋盘的四个层级
        for z in range(4):
            # 遍历每个层级的四行
            for y in range(4):
                # 打印空格，用于对齐
                print("   " * y, end="")
                # 遍历每行的四个位置
                for x in range(4):
                    # 获取指定位置的棋子颜色
                    p = board.get(x, y, z)
                    # 打印棋子的颜色和位置
                    print(f"({c[p.value]})      ", end="")
                # 换行
                print("\n")
            # 换行
            print("\n")
    # 定义一个名为 human_move 的方法，接受参数 self 和 board，并返回布尔值
    def human_move(self, board) -> bool:
        # 打印空行
        print()
        # 初始化字符串变量 c 为 "1234"
        c = "1234"
        # 进入无限循环
        while True:
            # 从用户输入中获取移动指令
            h = input("Your move?\n")
            # 如果用户输入为 "1"，返回 False
            if h == "1":
                return False
            # 如果用户输入为 "0"，显示游戏板并继续循环
            if h == "0":
                self.show_board(board)
                continue
            # 如果用户输入为三个字符且都在字符串 c 中
            if (len(h) == 3) and (h[0] in c) and (h[1] in c) and (h[2] in c):
                # 获取 x、y、z 坐标
                x = c.find(h[2])
                y = c.find(h[1])
                z = c.find(h[0])
                # 如果玩家移动成功，跳出循环
                if board.move_3d(x, y, z, Player.HUMAN):
                    break
                # 如果移动的方块已被使用，提示用户重试
                print("That square is used. Try again.")
            else:
                # 如果用户输入不符合要求，继续循环
                print("Incorrect move. Retype it--")  # 打印错误提示信息，要求重新输入

        return True  # 返回布尔值True

    def play(self) -> None:  # 定义play方法，不返回任何值
        print("Qubic\n")  # 打印游戏名称
        print("Create Computing Morristown, New Jersey\n\n\n")  # 打印游戏信息

        while True:  # 进入循环
            c = input("Do you want instructions?\n")  # 获取用户输入
            if len(c) >= 1 and (c[0] in "ynYN"):  # 判断用户输入是否符合要求
                break  # 符合要求则跳出循环
            print("Incorrect answer. Please type 'yes' or 'no.")  # 打印错误提示信息

        c = c.lower()  # 将用户输入转换为小写
        if c[0] == "y":  # 判断用户是否需要游戏说明
            print("The game is Tic-Tac-Toe in a 4 x 4 x 4 cube.")  # 打印游戏说明
            print("Each move is indicated by a 3 digit number, with each")  # 打印游戏说明
            print("digit between 1 and 4 inclusive.  The digits indicate the")  # 打印游戏说明
            print("level, row, and column, respectively, of the occupied")  # 打印游戏说明
            print("place.\n")  # 打印游戏说明
# 打印游戏板的提示信息
print("To print the playing board, type 0 (zero) as your move.")
print("The program will print the board with your moves indicated")
print("with a (Y), the machine's moves with an (M), and")
print("unused squares with a ( ).\n")

# 提示用户如何停止程序运行
print("To stop the program run, type 1 as your move.\n\n")

# 初始化游戏再玩一次的标志
play_again = True

# 循环进行游戏
while play_again:
    # 创建一个3D井字棋游戏板对象
    board = TicTacToe3D()

    # 循环直到用户输入正确的答案
    while True:
        s = input("Do you want to move first?\n")
        if len(s) >= 1 and (s[0] in "ynYN"):
            break
        print("Incorrect answer. Please type 'yes' or 'no'.")

    # 如果用户选择不先走，则跳过用户的回合
    skip_human = s[0] in "nN"
            move_text = [
                "Machine moves to",  # 机器移动到
                "Machine likes",  # 机器喜欢
                "Machine takes",  # 机器取走
                "Let's see you get out of this:  Machine moves to",  # 看看你能从这里走开：机器移动到
                "You fox.  Just in the nick of time, machine moves to",  # 你这家伙。正好及时，机器移动到
                "Nice try. Machine moves to",  # 不错的尝试。机器移动到
            ]

            while True:  # 无限循环
                if not skip_human and not self.human_move(board):  # 如果不跳过人类移动且人类移动失败
                    break  # 退出循环
                skip_human = False  # 重置跳过人类移动的标志

                m = board.machine_move()  # 获取机器移动
                assert m is not None  # 断言机器移动不为空
                if m[0] == Move.HUMAN_WIN:  # 如果机器移动导致人类获胜
                    print("You win as follows,")  # 打印人类获胜
                    self.show_win(board, m[2])  # type: ignore  # 显示人类获胜的情况
                    break  # 退出循环
                elif m[0] == Move.MACHINE_WIN:  # 如果机器赢了
                    print(  # 打印机器移动到的位置，并展示赢棋的方式
                        "Machine moves to {}, and wins as follows".format(
                            self.move_code(board, m[1])
                        )
                    )
                    self.show_win(board, m[2])  # type: ignore  # 展示赢棋的方式
                    break  # 结束游戏
                elif m[0] == Move.DRAW:  # 如果是平局
                    print("The game is a draw.")  # 打印平局信息
                    break  # 结束游戏
                elif m[0] == Move.CONCEDES:  # 如果机器认输
                    print("Machine concedes this game.")  # 打印机器认输信息
                    break  # 结束游戏
                else:  # 如果以上情况都不是
                    print(move_text[m[0].value - Move.MOVES.value])  # 打印移动的文本信息
                    print(self.move_code(board, m[1]))  # 打印机器移动的位置
                    board.move(m[1], Player.MACHINE)  # 机器移动
                self.show_board(board)  # 展示当前棋盘状态
            print(" ")  # 打印空行
            while True:  # 进入无限循环
                x = input("Do you want to try another game\n")  # 获取用户输入，询问是否想尝试另一个游戏
                if len(x) >= 1 and x[0] in "ynYN":  # 检查用户输入是否至少包含一个字符，并且第一个字符是'y'、'n'、'Y'或'N'
                    break  # 如果是，则跳出循环
                print("Incorrect answer. Please Type 'yes' or 'no'.")  # 如果用户输入不符合要求，则打印错误提示

            play_again = x[0] in "yY"  # 将用户输入的第一个字符是否为'y'或'Y'的布尔值赋给变量play_again


if __name__ == "__main__":  # 如果当前脚本被直接执行
    game = Qubit()  # 创建Qubit类的实例对象game
    game.play()  # 调用实例对象game的play方法
```