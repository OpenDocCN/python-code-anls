# `89_Tic-Tac-Toe\python\tictactoe2.py`

```
#!/usr/bin/env python3  # 指定使用 Python3 解释器来执行该脚本

from enum import Enum  # 导入 Enum 类，用于创建枚举类型


class OccupiedBy(Enum):  # 定义 OccupiedBy 枚举类型
    COMPUTER = -1  # 枚举值 COMPUTER 的值为 -1
    EMPTY = 0  # 枚举值 EMPTY 的值为 0
    PLAYER = 1  # 枚举值 PLAYER 的值为 1


class Winner(Enum):  # 定义 Winner 枚举类型
    NONE = 0  # 枚举值 NONE 的值为 0
    COMPUTER = 1  # 枚举值 COMPUTER 的值为 1
    PLAYER = 2  # 枚举值 PLAYER 的值为 2
    DRAW = 3  # 枚举值 DRAW 的值为 3


class Space(Enum):  # 定义 Space 枚举类型
    TOP_LEFT = 0  # 枚举值 TOP_LEFT 的值为 0
    TOP_CENTER = 1  # 枚举值 TOP_CENTER 的值为 1
    # 定义常量，表示棋盘上的位置
    TOP_RIGHT = 2
    MID_LEFT = 3
    MID_CENTER = 4
    MID_RIGHT = 5
    BOT_LEFT = 6
    BOT_CENTER = 7
    BOT_RIGHT = 8


# 定义函数 line_170，用于判断棋盘上的某些位置是否满足特定条件
def line_170(board, g, h, j, k):
    # 如果位置 g 被玩家占据，并且中心位置也被玩家占据
    if g == OccupiedBy.Player and board[Space.MID_CENTER] == g:
        # 如果顶部右侧位置也被玩家占据，并且底部左侧位置为空
        if (
            board[Space.TOP_RIGHT] == g and board[Space.BOTTOM_LEFT] is OccupiedBy.EMPTY
        ):  # Line 171
            return Space.BOTTOM_LEFT  # Line 187
        # 如果底部右侧位置也被玩家占据，并且顶部左侧位置为空
        elif (
            board[Space.BOTTOM_RIGHT] == g and board[Space.TOP_LEFT] is OccupiedBy.EMPTY
        ):  # Line 172
            return Space.TOP_LEFT  # Line 181
        # 其他情况...
def line_150(board, g, h, j, k):
    # 如果当前位置不是玩家的棋子
    if board[k] != g:  # line 150
        # 如果当前位置是计算机的棋子，或者当前位置下方或右侧的位置不是玩家的棋子
        if (
            board[k] == h  # line 160
            or board[k + 6] != g  # line 161
            or board[k + 3] != g
        ):  # line 162
            # 返回-1，跳转到170行
            return -1  # Goto 170
        else:
            return k + 3  # 返回 k + 3，表示函数执行结果为 k + 3  # Line 163
    elif board[k + 6] != g:  # 如果 board[k + 6] 不等于 g，则执行下面的条件判断  # line 152
        if board[k + 6] != 0 or board[k + 3] != g:  # 如果 board[k + 6] 不等于 0 或者 board[k + 3] 不等于 g，则执行下面的条件判断  # line 165
            return -1  # 返回 -1，表示函数执行结果为 -1  # Goto 170
    elif board[k + 3]:  # 如果 board[k + 3] 不为 0，则执行下面的条件判断  # line 156
        return -1  # 返回 -1，表示函数执行结果为 -1

    return k + 6  # 返回 k + 6，表示函数执行结果为 k + 6


def line_120(board, g, h, j, k):
    if board[j] != g:  # 如果 board[j] 不等于 g，则执行下面的条件判断
        if board[j] == h or board[j + 2] != g or board[j + 1] != g:  # 如果 board[j] 等于 h 或者 board[j + 2] 不等于 g 或者 board[j + 1] 不等于 g，则执行下面的条件判断
            if board[k] != g:  # 如果 board[k] 不等于 g，则执行下面的条件判断
                if board[k + 6] != g and (board[k + 6] != 0 or board[k + 3] != g):  # 如果 board[k + 6] 不等于 g 并且 (board[k + 6] 不等于 0 或者 board[k + 3] 不等于 g)，则执行下面的条件判断
                    # 450 IF G=1 THEN 465
                    pass  # 什么也不做
            elif board[j + 2] is not g:  # 如果 board[j + 2] 不是 g，则执行下面的条件判断  # Line 122
                pass  # 什么也不做
            elif board[j + 1] is not OccupiedBy.EMPTY:
                # 如果下一个位置不为空，则不做任何操作，直接跳过
                pass


def line_118(board, g, h):
    for j in range(7):
        for k in range(3):
            # 调用line_120函数，并返回其结果
            return line_120(board, g, h, j, k)


def think(board, g, h, moves):

    if board[Space.MID_CENTER] is OccupiedBy.EMPTY:
        # 如果中心位置为空，则返回中心位置
        return Space.MID_CENTER

    if board[Space.MID_CENTER] is OccupiedBy.PLAYER:
        if (
            board[Space.TOP_CENTER] is OccupiedBy.PLAYER
            and board[Space.TOP_LEFT] is OccupiedBy.EMPTY
            or board[Space.MID_LEFT] is OccupiedBy.PLAYER
```
在这段代码中，需要添加注释来解释每个语句的作用和功能。
# 检查条件，如果满足则返回对应的空格位置
if (
    board[Space.TOP_LEFT] is OccupiedBy.PLAYER
    and board[Space.MID_LEFT] is OccupiedBy.EMPTY
    or board[Space.MID_CENTER] is OccupiedBy.PLAYER
    and board[Space.MID_LEFT] is OccupiedBy.EMPTY
):
    return Space.MID_LEFT
elif (
    board[Space.TOP_RIGHT] is OccupiedBy.PLAYER
    and board[Space.BOT_RIGHT] is OccupiedBy.EMPTY
    or board[Space.BOT_CENTER] is OccupiedBy.PLAYER
    and board[Space.BOT_RIGHT] is OccupiedBy.EMPTY
):
    return Space.BOT_RIGHT

# 检查条件，如果满足则返回对应的空格位置
if g == OccupiedBy.PLAYER:
    j = 3 * int((moves - 1) / 3)
    if move == j + 1:  # noqa: This definitely is a bug!
        k = 1
    if move == j + 2:  # noqa: This definitely is a bug!
        k = 2
    if move == j + 3:  # noqa: This definitely is a bug!
        k = 3
    return subthink(g, h, j, k)  # noqa: This definitely is a bug!
def render_board(board, space_mapping):
    # 定义垂直分隔符
    vertical_divider = "!"
    # 定义水平分隔符
    horizontal_divider = "---+---+---"
    # 创建空列表用于存储每行的内容
    lines = []
    # 将每行的棋盘空间映射为字符串，并添加到列表中
    lines.append(vertical_divider.join(space_mapping[space] for space in board[0:3]))
    # 添加水平分隔符到列表中
    lines.append(horizontal_divider)
    # 将每行的棋盘空间映射为字符串，并添加到列表中
    lines.append(vertical_divider.join(space_mapping[space] for space in board[3:6]))
    # 添加水平分隔符到列表中
    lines.append(horizontal_divider)
    # 将每行的棋盘空间映射为字符串，并添加到列表中
    lines.append(vertical_divider.join(space_mapping[space] for space in board[6:9]))
    # 将列表中的内容用换行符连接成一个字符串并返回
    return "\n".join(lines)


def determine_winner(board, g):
    # 检查是否有匹配的水平线
    for i in range(Space.TOP_LEFT.value, Space.BOT_LEFT.value + 1, 3):  # 遍历每一行的起始位置
        if board[i] != board[i + 1] or board[i] != board[i + 2]:  # 检查每一行是否有相同的棋子
            continue  # 继续下一次循环
        elif board[i] == OccupiedBy.COMPUTER:  # 如果当前行的棋子都是电脑的，则...
    # Check for matching vertical lines
    for i in range(
        Space.TOP_LEFT.value, Space.TOP_RIGHT.value + 1, 1
    ):  # 从左上角到右上角遍历每一列
        if (
            board[i] != board[i + 3] or board[i] != board[i + 6]
        ):  # 检查每一列上是否有连续三个相同的棋子
            continue  # 如果没有，则继续下一列
        elif board[i] == OccupiedBy.COMPUTER:  # 如果有连续三个计算机的棋子
            return Winner.COMPUTER  # 则计算机获胜
        elif board[i] == OccupiedBy.PLAYER:  # 如果有连续三个玩家的棋子
            return Winner.PLAYER  # 则玩家获胜

    # Check diagonals
    if any(space is OccupiedBy.EMPTY for space in board):  # 检查棋盘上是否还有空位
        if board[Space.MID_CENTER.value] != g:  # 如果中心位置不是空位
        return Winner.NONE  # 如果没有玩家获胜，返回Winner.NONE
        elif (
            board[Space.TOP_LEFT.value] == g and board[Space.BOT_RIGHT.value] == g
        ) or (board[Space.BOT_LEFT.value] == g and board[Space.TOP_RIGHT.value] == g):
            return Winner.COMPUTER if g is OccupiedBy.COMPUTER else Winner.PLAYER  # 如果玩家或计算机获胜，返回Winner.COMPUTER或Winner.PLAYER
        else:
            return Winner.NONE  # 如果没有玩家获胜，返回Winner.NONE

    return Winner.DRAW  # 如果没有玩家或计算机获胜，返回Winner.DRAW，表示平局


def computer_think(board):
    empty_indices = [
        index for index, space in enumerate(board) if space is OccupiedBy.EMPTY  # 找出棋盘中空位置的索引
    ]

    return empty_indices[0]  # 返回第一个空位置的索引


def prompt_player(board):
    while True:  # 进入无限循环，直到条件满足跳出循环
        move = int(input("\nWHERE DO YOU MOVE? "))  # 获取用户输入的移动位置并转换为整数

        if move == 0:  # 如果用户输入为0
            return 0  # 返回0

        if move > 9 or board[move - 1] is not OccupiedBy.EMPTY:  # 如果用户输入大于9或者所选位置已经被占据
            print("THAT SQUARE IS OCCUPIED.\n\n")  # 打印提示信息
            continue  # 继续下一次循环

        return move  # 返回用户输入的移动位置


def main() -> None:  # 定义一个名为main的函数，不返回任何值
    print(" " * 30 + "TIC-TAC-TOE")  # 打印TIC-TAC-TOE
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 打印CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY
    print("\n\n")  # 打印两个换行

    print("THE BOARD IS NUMBERED:")  # 打印提示信息
    print(" 1  2  3")  # 打印数字1、2、3
    # 打印游戏板的初始状态
    print(" 1 | 2 | 3 ")
    print("-----------")
    print(" 4 | 5 | 6 ")
    print("-----------")
    print(" 7 | 8 | 9 ")
    print("\n\n")

    # 创建一个包含9个空格的列表，表示游戏板的状态
    board = [OccupiedBy.EMPTY] * 9
    # 设置当前玩家为玩家
    current_player = OccupiedBy.PLAYER
    # 创建一个字典，将不同状态对应的符号进行映射
    space_mapping = {
        OccupiedBy.EMPTY: "   ",
        OccupiedBy.PLAYER: " X ",
        OccupiedBy.COMPUTER: " O ",
    }

    # 获取玩家选择的符号
    symbol = input("DO YOU WANT 'X' OR 'O'? ").upper()

    # 如果玩家选择的不是X，则默认为O，并且电脑先手
    if symbol != "X":
        space_mapping[OccupiedBy.PLAYER] = " O "
        space_mapping[OccupiedBy.COMPUTER] = " X "
        current_player = OccupiedBy.COMPUTER  # 设置当前玩家为计算机

    while True:  # 进入游戏循环
        if current_player is OccupiedBy.PLAYER:  # 如果当前玩家是玩家
            move = prompt_player(board)  # 提示玩家输入移动位置
            if move == 0:  # 如果玩家输入0
                print("THANKS FOR THE GAME.")  # 打印感谢信息
                break  # 退出游戏循环
            board[move - 1] = current_player  # 在游戏板上标记玩家的移动位置

        elif current_player is OccupiedBy.COMPUTER:  # 如果当前玩家是计算机
            print("\nTHE COMPUTER MOVES TO...")  # 打印计算机移动信息
            board[computer_think(board)] = current_player  # 计算机根据算法选择移动位置

        print(render_board(board, space_mapping))  # 打印游戏板

        winner = determine_winner(board, current_player)  # 判断是否有玩家获胜

        if winner is not Winner.NONE:  # 如果有玩家获胜
            print(winner)  # 打印获胜信息
            break  # 结束循环，跳出当前循环

        if current_player is OccupiedBy.COMPUTER:  # 如果当前玩家是电脑
            current_player = OccupiedBy.PLAYER  # 则将当前玩家设为玩家
        elif current_player is OccupiedBy.PLAYER:  # 如果当前玩家是玩家
            current_player = OccupiedBy.COMPUTER  # 则将当前玩家设为电脑


if __name__ == "__main__":  # 如果当前脚本被直接执行
    main()  # 调用主函数进行程序执行
```