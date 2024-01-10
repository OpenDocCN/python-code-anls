# `basic-computer-games\89_Tic-Tac-Toe\python\tictactoe2.py`

```
#!/usr/bin/env python3
# 导入枚举类型模块
from enum import Enum

# 定义棋盘上的占用状态枚举
class OccupiedBy(Enum):
    COMPUTER = -1
    EMPTY = 0
    PLAYER = 1

# 定义游戏结果的枚举
class Winner(Enum):
    NONE = 0
    COMPUTER = 1
    PLAYER = 2
    DRAW = 3

# 定义棋盘上的空间位置枚举
class Space(Enum):
    TOP_LEFT = 0
    TOP_CENTER = 1
    TOP_RIGHT = 2
    MID_LEFT = 3
    MID_CENTER = 4
    MID_RIGHT = 5
    BOT_LEFT = 6
    BOT_CENTER = 7
    BOT_RIGHT = 8

# 定义函数 line_170
def line_170(board, g, h, j, k):
    # 如果当前位置被玩家占据，并且中心位置也被玩家占据
    if g == OccupiedBy.Player and board[Space.MID_CENTER] == g:
        # 如果右上角位置也被玩家占据，并且左下角位置为空
        if (
            board[Space.TOP_RIGHT] == g and board[Space.BOTTOM_LEFT] is OccupiedBy.EMPTY
        ):  # Line 171
            return Space.BOTTOM_LEFT  # Line 187
        # 如果右下角位置也被玩家占据，并且左上角位置为空
        elif (
            board[Space.BOTTOM_RIGHT] == g and board[Space.TOP_LEFT] is OccupiedBy.EMPTY
        ):  # Line 172
            return Space.TOP_LEFT  # Line 181
        # 如果左下角位置也被玩家占据，并且右上角位置为空，或者右下角位置被玩家占据并且右上角位置为空
        elif (
            board[Space.BOTTOM_LEFT] == g and board[Space.TOP_RIGHT] is OccupiedBy.EMPTY
        ) or (
            board[Space.BOTTOM_RIGHT] is OccupiedBy.PLAYER
            and board[Space.TOP_RIGHT] is OccupiedBy.EMPTY
        ):  # Line 173 and 174
            return Space.TOP_RIGHT  # Line 183 and Line 189
        # 如果当前位置被计算机占据
        elif g is OccupiedBy.COMPUTER:
            # 修改当前位置为玩家占据，中心位置为计算机占据
            g = OccupiedBy.PLAYER
            h = OccupiedBy.COMPUTER
            # 调用 line_118 函数
            return line_118(board, g, h, j, k)

# 定义函数 line_150
def line_150(board, g, h, j, k):
    # 如果当前位置不被占据
    if board[k] != g:  # line 150
        # 如果当前位置被对手占据，或者下方位置不被当前玩家占据，或者右侧位置不被当前玩家占据
        if (
            board[k] == h  # line 160
            or board[k + 6] != g  # line 161
            or board[k + 3] != g
        ):  # line 162
            return -1  # Goto 170
        else:
            return k + 3  # Line 163
    # 如果当前位置被占据，并且下方位置不被当前玩家占据
    elif board[k + 6] != g:  # line 152
        # 如果下方位置不为空，或者右侧位置不被当前玩家占据
        if board[k + 6] != 0 or board[k + 3] != g:  # line 165
            return -1  # Goto 170
    # 如果右侧位置被占据
    elif board[k + 3]:  # line 156
        return -1

    return k + 6

# 定义函数 line_120
def line_120(board, g, h, j, k):
    # 如果当前位置的值不等于g
    if board[j] != g:
        # 如果当前位置的值等于h，或者当前位置后两个位置的值不等于g，或者当前位置后一个位置的值不等于g
        if board[j] == h or board[j + 2] != g or board[j + 1] != g:
            # 如果当前位置后一个位置的值不等于g
            if board[k] != g:
                # 如果当前位置后六个位置的值不等于g，并且（当前位置后六个位置的值不等于0，或者当前位置后三个位置的值不等于g）
                if board[k + 6] != g and (board[k + 6] != 0 or board[k + 3] != g):
                    # 450 IF G=1 THEN 465
                    pass
            # 如果当前位置后两个位置的值不等于g
            elif board[j + 2] is not g:  # Line 122
                pass
            # 如果当前位置后一个位置的值不等于空
            elif board[j + 1] is not OccupiedBy.EMPTY:
                pass
# 定义函数 line_118，接受三个参数 board, g, h
def line_118(board, g, h):
    # 循环遍历 j 取值范围为 0 到 6
    for j in range(7):
        # 循环遍历 k 取值范围为 0 到 2
        for k in range(3):
            # 调用函数 line_120，并返回结果
            return line_120(board, g, h, j, k)

# 定义函数 think，接受四个参数 board, g, h, moves
def think(board, g, h, moves):
    # 如果中心位置为空，则返回中心位置
    if board[Space.MID_CENTER] is OccupiedBy.EMPTY:
        return Space.MID_CENTER
    # 如果中心位置被玩家占据
    if board[Space.MID_CENTER] is OccupiedBy.PLAYER:
        # 如果上中位置被玩家占据并且上左位置为空，或者中左位置被玩家占据并且上左位置为空，则返回左下位置
        if (
            board[Space.TOP_CENTER] is OccupiedBy.PLAYER
            and board[Space.TOP_LEFT] is OccupiedBy.EMPTY
            or board[Space.MID_LEFT] is OccupiedBy.PLAYER
            and board[Space.TOP_LEFT] is OccupiedBy.EMPTY
        ):
            return Space.BOT_LEFT
        # 如果中右位置被玩家占据并且下右位置为空，或者下中位置被玩家占据并且下右位置为空，则返回右下位置
        elif (
            board[Space.MID_RIGHT] is OccupiedBy.PLAYER
            and board[Space.BOT_RIGHT] is OccupiedBy.EMPTY
            or board[Space.BOT_CENTER] is OccupiedBy.PLAYER
            and board[Space.BOT_RIGHT] is OccupiedBy.EMPTY
        ):
            return Space.BOT_RIGHT
    # 如果 g 等于玩家
    if g == OccupiedBy.PLAYER:
        # 计算 j 的值
        j = 3 * int((moves - 1) / 3)
        # 如果 move 等于 j + 1，则 k 等于 1
        if move == j + 1:  # noqa: This definitely is a bug!
            k = 1
        # 如果 move 等于 j + 2，则 k 等于 2
        if move == j + 2:  # noqa: This definitely is a bug!
            k = 2
        # 如果 move 等于 j + 3，则 k 等于 3
        if move == j + 3:  # noqa: This definitely is a bug!
            k = 3
        # 调用函数 subthink，并返回结果
        return subthink(g, h, j, k)  # noqa: This definitely is a bug!

# 定义函数 render_board，接受两个参数 board, space_mapping
def render_board(board, space_mapping):
    # 定义垂直分隔符
    vertical_divider = "!"
    # 定义水平分隔符
    horizontal_divider = "---+---+---"
    # 定义空列表 lines
    lines = []
    # 将第一行的空格映射值连接起来，添加到 lines 列表中
    lines.append(vertical_divider.join(space_mapping[space] for space in board[0:3]))
    # 添加水平分隔符到 lines 列表中
    lines.append(horizontal_divider)
    # 将第二行的空格映射值连接起来，添加到 lines 列表中
    lines.append(vertical_divider.join(space_mapping[space] for space in board[3:6]))
    # 添加水平分隔符到 lines 列表中
    lines.append(horizontal_divider)
    # 将第三行的空格映射值连接起来，添加到 lines 列表中
    lines.append(vertical_divider.join(space_mapping[space] for space in board[6:9]))
    # 返回 lines 列表中的元素，用换行符连接起来
    return "\n".join(lines)

# 定义函数 determine_winner，接受两个参数 board, g
def determine_winner(board, g):
    # 检查是否有匹配的水平线
    # 遍历以3为步长的水平线上的格子，从左上角到左下角
    for i in range(Space.TOP_LEFT.value, Space.BOT_LEFT.value + 1, 3):  # Line 1095
        # 如果当前格子与其右边两个格子的值不相等，则跳过
        if board[i] != board[i + 1] or board[i] != board[i + 2]:  # Lines 1100 and 1105
            continue  # First third of Line 1115
        # 如果当前格子被计算机占据，则返回计算机获胜
        elif board[i] == OccupiedBy.COMPUTER:  #
            return Winner.COMPUTER
        # 如果当前格子被玩家占据，则返回玩家获胜
        elif board[i] == OccupiedBy.PLAYER:
            return Winner.PLAYER

    # 检查垂直线上的匹配
    for i in range(
        Space.TOP_LEFT.value, Space.TOP_RIGHT.value + 1, 1
    ):  # Second third of Line 1115
        # 如果当前格子与其下方两个格子的值不相等，则跳过
        if (
            board[i] != board[i + 3] or board[i] != board[i + 6]
        ):  # Last third of Line 1115
            continue  # First third of 1150
        # 如果当前格子被计算机占据，则返回计算机获胜
        elif board[i] == OccupiedBy.COMPUTER:  # Line 1135
            return Winner.COMPUTER
        # 如果当前格子被玩家占据，则返回玩家获胜
        elif board[i] == OccupiedBy.PLAYER:  # Line 1137
            return Winner.PLAYER

    # 检查对角线
    if any(space is OccupiedBy.EMPTY for space in board):
        # 如果棋盘中有空格子，则游戏尚未结束
        if board[Space.MID_CENTER.value] != g:
            return Winner.NONE
        # 如果中心格子被占据，并且两条对角线上的格子被同一方占据，则返回对应的获胜方
        elif (
            board[Space.TOP_LEFT.value] == g and board[Space.BOT_RIGHT.value] == g
        ) or (board[Space.BOT_LEFT.value] == g and board[Space.TOP_RIGHT.value] == g):
            return Winner.COMPUTER if g is OccupiedBy.COMPUTER else Winner.PLAYER
        else:
            return Winner.NONE
    # 如果棋盘已满且无一方获胜，则返回平局
    return Winner.DRAW
# 计算机思考函数，返回第一个空位置的索引
def computer_think(board):
    # 找出所有空位置的索引
    empty_indices = [
        index for index, space in enumerate(board) if space is OccupiedBy.EMPTY
    ]

    return empty_indices[0]


# 提示玩家函数，要求玩家输入移动位置
def prompt_player(board):
    while True:
        move = int(input("\nWHERE DO YOU MOVE? "))

        if move == 0:
            return 0

        # 如果玩家选择的位置超出范围或者已经被占据，则提示并继续循环
        if move > 9 or board[move - 1] is not OccupiedBy.EMPTY:
            print("THAT SQUARE IS OCCUPIED.\n\n")
            continue

        return move


# 主函数
def main() -> None:
    print(" " * 30 + "TIC-TAC-TOE")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n\n")

    print("THE BOARD IS NUMBERED:")
    print(" 1  2  3")
    print(" 4  5  6")
    print(" 7  8  9")
    print("\n\n")

    # 默认状态下，创建一个包含9个空位置的列表
    board = [OccupiedBy.EMPTY] * 9
    # 当前玩家默认为玩家
    current_player = OccupiedBy.PLAYER
    # 空位置和玩家/计算机的映射关系
    space_mapping = {
        OccupiedBy.EMPTY: "   ",
        OccupiedBy.PLAYER: " X ",
        OccupiedBy.COMPUTER: " O ",
    }

    # 要求玩家选择 'X' 或 'O'
    symbol = input("DO YOU WANT 'X' OR 'O'? ").upper()

    # 如果玩家不选择 'X'，则默认为 'O'，并且计算机先手
    if symbol != "X":
        space_mapping[OccupiedBy.PLAYER] = " O "
        space_mapping[OccupiedBy.COMPUTER] = " X "
        current_player = OccupiedBy.COMPUTER
    # 无限循环，直到游戏结束
    while True:
        # 如果当前玩家是玩家
        if current_player is OccupiedBy.PLAYER:
            # 提示玩家输入移动位置
            move = prompt_player(board)
            # 如果移动位置为0，结束游戏
            if move == 0:
                print("THANKS FOR THE GAME.")
                break
            # 在棋盘上标记当前玩家的移动位置
            board[move - 1] = current_player

        # 如果当前玩家是电脑
        elif current_player is OccupiedBy.COMPUTER:
            # 提示电脑移动
            print("\nTHE COMPUTER MOVES TO...")
            # 计算电脑的移动位置并在棋盘上标记
            board[computer_think(board)] = current_player

        # 打印当前棋盘状态
        print(render_board(board, space_mapping))

        # 判断是否有玩家获胜
        winner = determine_winner(board, current_player)

        # 如果有玩家获胜，打印获胜信息并结束游戏
        if winner is not Winner.NONE:
            print(winner)
            break

        # 切换当前玩家
        if current_player is OccupiedBy.COMPUTER:
            current_player = OccupiedBy.PLAYER
        elif current_player is OccupiedBy.PLAYER:
            current_player = OccupiedBy.COMPUTER
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```