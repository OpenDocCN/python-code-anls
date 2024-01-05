# `d:/src/tocomm/basic-computer-games\72_Queen\python\queen.py`

```
#!/usr/bin/env python3
"""
Implementation of Queens game in Python 3.

Original game in BASIC by David Ahl in _BASIC Comuter Games_, published in 1978,
as reproduced here:
    https://www.atariarchives.org/basicgames/showpage.php?page=133

Port to Python 3 by Christopher L. Phan <https://chrisphan.com>

Supports Python version 3.8 or later.
"""

from random import random  # 从 random 模块中导入 random 函数
from typing import Final, FrozenSet, Optional, Tuple  # 从 typing 模块中导入 Final, FrozenSet, Optional, Tuple 类型

########################################################################################
#                                  Optional configs
########################################################################################
# You can edit these variables to change the behavior of the game.
# The original implementation has a bug that allows a player to move off the board,
# e.g. start at the nonexistant space 91. Change the variable FIX_BOARD_BUG to ``True``
# to fix this behavior.
# 原始实现存在一个bug，允许玩家移出棋盘，例如从不存在的空间91开始。将变量FIX_BOARD_BUG更改为``True``以修复此行为。

FIX_BOARD_BUG: Final[bool] = False

# In the original implementation, the board is only printed once. Change the variable
# SHOW_BOARD_ALWAYS to ``True`` to display the board every time.
# 在原始实现中，棋盘只打印一次。将变量SHOW_BOARD_ALWAYS更改为``True``以每次显示棋盘。

SHOW_BOARD_ALWAYS: Final[bool] = False

# In the original implementaiton, the board is printed a bit wonky because of the
# differing widths of the numbers. Change the variable ALIGNED_BOARD to ``True`` to
# fix this.
# 在原始实现中，由于数字宽度不同，棋盘打印有点奇怪。将变量ALIGNED_BOARD更改为``True``以修复这个问题。

ALIGNED_BOARD: Final[bool] = False

########################################################################################
# 定义一个名为INSTR_TXT的常量，类型为字符串，存储游戏的说明文本
INSTR_TXT: Final[str] = """WE ARE GOING TO PLAY A GAME BASED ON ONE OF THE CHESS
MOVES.  OUR QUEEN WILL BE ABLE TO MOVE ONLY TO THE LEFT,
DOWN, OR DIAGONALLY DOWN AND TO THE LEFT.

THE OBJECT OF THE GAME IS TO PLACE THE QUEEN IN THE LOWER
LEFT HAND SQUARE BY ALTERNATING MOVES BETWEEN YOU AND THE
COMPUTER.  THE FIRST ONE TO PLACE THE QUEEN THERE WINS.

YOU GO FIRST AND PLACE THE QUEEN IN ANY ONE OF THE SQUARES
ON THE TOP ROW OR RIGHT HAND COLUMN.
THAT WILL BE YOUR FIRST MOVE.
WE ALTERNATE MOVES.
YOU MAY FORFEIT BY TYPING '0' AS YOUR MOVE.
BE SURE TO PRESS THE RETURN KEY AFTER EACH RESPONSE.

"""
WIN_MSG: Final[str] = """C O N G R A T U L A T I O N S . . .  # 赢得游戏的消息
YOU HAVE WON--VERY WELL PLAYED.  # 赢得游戏的祝贺消息
IT LOOKS LIKE I HAVE MET MY MATCH.  # 表示对手很强大
THANKS FOR PLAYING---I CAN'T WIN ALL THE TIME.  # 感谢对手参与游戏

"""

LOSE_MSG: Final[str] = """
NICE TRY, BUT IT LOOKS LIKE I HAVE WON.  # 输掉游戏的消息
THANKS FOR PLAYING.  # 感谢对手参与游戏

"""
def loc_to_num(location: Tuple[int, int], fix_align: bool = False) -> str:
    """Convert a position given by row, column into a space number."""
    # 定义一个函数，将给定的行和列位置转换为空格号
    row, col = location
    # 将行和列位置转换为空格号的字符串
    out_str: str = f"{row + 8 - col}{row + 1}"
    # 如果不需要修正对齐或者字符串长度为3，则直接返回转换后的字符串
    if not fix_align or len(out_str) == 3:
        return out_str
    else:
        # 否则在字符串末尾添加一个空格并返回
        return out_str + " "


GAME_BOARD: Final[str] = (
    "\n"
    + "\n\n\n".join(
        # 生成游戏棋盘的字符串表示，使用loc_to_num函数将每个位置转换为空格号
        "".join(f" {loc_to_num((row, col), ALIGNED_BOARD)} " for col in range(8))
        for row in range(8)
    )
    + "\n\n\n"
)
def num_to_loc(num: int) -> Tuple[int, int]:
    """Convert a space number into a position given by row, column."""
    # 将数字转换为行和列的位置
    row: int = num % 10 - 1  # 计算行数
    col: int = row + 8 - (num - row - 1) // 10  # 计算列数
    return row, col  # 返回行和列的位置


# The win location
WIN_LOC: Final[Tuple[int, int]] = (7, 0)  # 定义游戏胜利的位置为元组(7, 0)

# These are the places (other than the win condition) that the computer will always
# try to move into.
COMPUTER_SAFE_SPOTS: Final[FrozenSet[Tuple[int, int]]] = frozenset(
    [
        (2, 3),
        (4, 5),
        (5, 1),
        (6, 2),
    ]
)  # 定义计算机总是尝试移动到的安全位置的集合
# These are the places that the computer will always try to move into.
# 计算机将始终尝试移动到这些位置。
COMPUTER_PREF_MOVES: Final[
    FrozenSet[Tuple[int, int]]
] = COMPUTER_SAFE_SPOTS | frozenset([WIN_LOC])

# These are the locations (not including the win location) from which either player can
# force a win (but the computer will always choose one of the COMPUTER_PREF_MOVES).
# 这些位置（不包括获胜位置）是任一玩家可以强制获胜的位置（但计算机将始终选择COMPUTER_PREF_MOVES中的一个）。
SAFE_SPOTS: Final[FrozenSet[Tuple[int, int]]] = COMPUTER_SAFE_SPOTS | frozenset(
    [
        (0, 4),
        (3, 7),
    ]
)

def intro() -> None:
    """Print the intro and print instructions if desired."""
    # 打印介绍并在需要时打印说明。
    print(" " * 33 + "Queen")
    print(" " * 15 + "Creative Computing  Morristown, New Jersey")
    print("\n" * 2)  # 打印两行空行
    if ask("DO YOU WANT INSTRUCTIONS"):  # 如果用户想要说明书
        print(INSTR_TXT)  # 打印说明书的文本内容


def get_move(current_loc: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    """Get the next move from the player."""  # 从玩家获取下一步移动
    prompt: str  # 提示信息
    player_resp: str  # 玩家的回应
    move_raw: int  # 原始移动
    new_row: int  # 新的行
    new_col: int  # 新的列
    if current_loc is None:  # 如果当前位置为空（第一轮）
        prompt = "WHERE WOULD YOU LIKE TO START? "  # 提示玩家选择起始位置
    else:
        prompt = "WHAT IS YOUR MOVE? "  # 提示玩家选择移动方向
        row, col = current_loc  # 获取当前位置的行和列
    while True:  # 无限循环
        player_resp = input(prompt).strip()  # 获取玩家的输入并去除首尾空格
        try:  # 尝试
            move_raw = int(player_resp)  # 将玩家输入的字符串转换为整数
            if move_raw == 0:  # 如果玩家选择放弃比赛
                return 8, 8  # 返回特定的行和列值
            new_row, new_col = num_to_loc(move_raw)  # 将玩家输入的整数转换为行和列的位置
            if current_loc is None:  # 如果当前位置为空
                if (new_row == 0 or new_col == 7) and (  # 如果新位置在边界上，并且不是固定棋盘错误或者在特定范围内
                    not FIX_BOARD_BUG or (new_col >= 0 and new_row < 8)
                ):
                    return new_row, new_col  # 返回新的行和列值
                else:
                    prompt = (  # 设置提示信息
                        "PLEASE READ THE DIRECTIONS AGAIN.\n"
                        "YOU HAVE BEGUN ILLEGALLY.\n\n"
                        "WHERE WOULD YOU LIKE TO START? "
                    )
            else:  # 如果当前位置不为空
                if (
                    (new_row == row and new_col < col)  # 如果新位置向左移动
                    or (new_col == col and new_row > row)  # 如果新位置向下移动
                    or (new_row - row == col - new_col)  # 如果新位置向左下对角线移动
def random_computer_move(location: Tuple[int, int]) -> Tuple[int, int]:
    """Make a random move."""
    # 定义一个函数，接受一个元组类型的参数，返回一个元组类型的值
    row, col = location
    # 从位置参数中获取行和列的值
    if (z := random()) > 0.6:
        # 如果随机数大于0.6
        # 向下移动一个空格
        return row + 1, col
    elif z > 0.3:
        # 如果随机数大于0.3
        # 向左下方移动一个空格
        return row + 1, col - 1
    else:
        # 否则
        # 向左移动一个空格
        return row, col - 1
# 返回当前位置的上一列

def computer_move(location: Tuple[int, int]) -> Tuple[int, int]:
    """Get the computer's move."""
    # 获取计算机的移动位置
    # 如果玩家已经做出了最佳移动，那么选择一个随机移动
    if location in SAFE_SPOTS:
        return random_computer_move(location)
    # 我们不需要实现检查玩家是否获胜的逻辑，因为在调用此函数之前已经检查过了。
    row, col = location
    for k in range(7, 0, -1):
        # 如果计算机可以向左移动 k 步并最终到达一个安全位置或获胜，那么就这样做。
        if (new_loc := (row, col - k)) in COMPUTER_PREF_MOVES:
            return new_loc
        # 如果计算机可以向下移动 k 步并最终到达一个安全位置或获胜，那么就这样做。
        if (new_loc := (row + k, col)) in COMPUTER_PREF_MOVES:
            return new_loc
        # 如果计算机可以对角线移动 k 步并最终到达一个安全位置或获胜，那么就这样做。
        # do it.
        # 如果新位置在计算机优先移动的列表中，则返回新位置
        if (new_loc := (row + k, col - k)) in COMPUTER_PREF_MOVES:
            return new_loc
        # 如果不在计算机优先移动的列表中，则执行随机移动。（注意：实际上不应该发生这种情况——如果玩家没有在安全位置中移动，应该总是能够进行最佳移动。）
    return random_computer_move(location)


def main_game() -> None:
    """Execute the main game."""
    game_over: bool = False
    location: Optional[Tuple[int, int]] = None  # Indicate it is the first turn
    while not game_over:
        location = get_move(location)
        if location == (8, 8):  # 当玩家输入0时返回(8, 8)，表示玩家认输
            print("\nIT LOOKS LIKE I HAVE WON BY FORFEIT.\n")
            game_over = True
        elif location == WIN_LOC:  # 玩家赢了（在左下角）
            print(WIN_MSG)
            game_over = True  # 设置游戏结束标志为真
        else:
            location = computer_move(location)  # 计算机移动到新位置
            print(f"COMPUTER MOVES TO SQUARE {loc_to_num(location)}")  # 打印计算机移动到的方块位置
            if location == WIN_LOC:  # 如果计算机赢了（在左下角）
                print(LOSE_MSG)  # 打印失败信息
                game_over = True  # 设置游戏结束标志为真
        # 默认行为是不在每个回合都显示游戏板，但可以通过在文件开头修改一个标志来修改这一行为
        if not game_over and SHOW_BOARD_ALWAYS:  # 如果游戏没有结束且总是显示游戏板的标志为真
            print(GAME_BOARD)  # 打印游戏板


def ask(prompt: str) -> bool:
    """Ask a yes/no question until user gives an understandable response."""
    inpt: str  # 输入变量的类型为字符串
    while True:
        # 将输入规范化为大写，去除空格，然后获取第一个字符
        inpt = input(prompt + "? ").upper().strip()[0]  # 获取用户输入并规范化
        print()  # 打印空行
        if inpt == "Y":  # 如果输入是"Y"，返回True
            return True
        elif inpt == "N":  # 如果输入是"N"，返回False
            return False
        print("PLEASE ANSWER 'YES' OR 'NO'.")  # 如果输入既不是"Y"也不是"N"，打印提示信息
    return False  # 默认返回False


if __name__ == "__main__":
    intro()  # 调用intro函数，介绍游戏
    still_playing: bool = True  # 初始化still_playing变量为True
    while still_playing:  # 当still_playing为True时，进入循环
        print(GAME_BOARD)  # 打印游戏板
        main_game()  # 调用main_game函数，进行游戏
        still_playing = ask("ANYONE ELSE CARE TO TRY")  # 调用ask函数询问是否还有人想尝试，将结果赋给still_playing
    print("\nOK --- THANKS AGAIN.")  # 循环结束后打印结束语
```