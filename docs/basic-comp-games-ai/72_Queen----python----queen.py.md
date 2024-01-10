# `basic-computer-games\72_Queen\python\queen.py`

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

from random import random
from typing import Final, FrozenSet, Optional, Tuple

########################################################################################
#                                  Optional configs
########################################################################################
# You can edit these variables to change the behavior of the game.
#
# The original implementation has a bug that allows a player to move off the board,
# e.g. start at the nonexistant space 91. Change the variable FIX_BOARD_BUG to ``True``
# to fix this behavior.
#

FIX_BOARD_BUG: Final[bool] = False

# In the original implementation, the board is only printed once. Change the variable
# SHOW_BOARD_ALWAYS to ``True`` to display the board every time.

SHOW_BOARD_ALWAYS: Final[bool] = False

# In the original implementaiton, the board is printed a bit wonky because of the
# differing widths of the numbers. Change the variable ALIGNED_BOARD to ``True`` to
# fix this.

ALIGNED_BOARD: Final[bool] = False

########################################################################################

INSTR_TXT: Final[
    str
] = """WE ARE GOING TO PLAY A GAME BASED ON ONE OF THE CHESS
MOVES.  OUR QUEEN WILL BE ABLE TO MOVE ONLY TO THE LEFT,
DOWN, OR DIAGONALLY DOWN AND TO THE LEFT.

THE OBJECT OF THE GAME IS TO PLACE THE QUEEN IN THE LOWER
LEFT HAND SQUARE BY ALTERNATING MOVES BETWEEN YOU AND THE
COMPUTER.  THE FIRST ONE TO PLACE THE QUEEN THERE WINS.

YOU GO FIRST AND PLACE THE QUEEN IN ANY ONE OF THE SQUARES
ON THE TOP ROW OR RIGHT HAND COLUMN.
THAT WILL BE YOUR FIRST MOVE.
WE ALTERNATE MOVES.
# 你可能会输掉游戏，如果你在你的移动中输入'0'。
# 确保在每次回应后按下回车键。

"""

# 赢得游戏的消息
WIN_MSG: Final[str] = """C O N G R A T U L A T I O N S . . .

YOU HAVE WON--VERY WELL PLAYED.
IT LOOKS LIKE I HAVE MET MY MATCH.
THANKS FOR PLAYING---I CAN'T WIN ALL THE TIME.

"""

# 输掉游戏的消息
LOSE_MSG: Final[str] = """
NICE TRY, BUT IT LOOKS LIKE I HAVE WON.
THANKS FOR PLAYING.

"""


def loc_to_num(location: Tuple[int, int], fix_align: bool = False) -> str:
    """将由行、列给出的位置转换为空格号码。"""
    row, col = location
    out_str: str = f"{row + 8 - col}{row + 1}"
    if not fix_align or len(out_str) == 3:
        return out_str
    else:
        return out_str + " "


# 游戏板
GAME_BOARD: Final[str] = (
    "\n"
    + "\n\n\n".join(
        "".join(f" {loc_to_num((row, col), ALIGNED_BOARD)} " for col in range(8))
        for row in range(8)
    )
    + "\n\n\n"
)


def num_to_loc(num: int) -> Tuple[int, int]:
    """将空格号码转换为由行、列给出的位置。"""
    row: int = num % 10 - 1
    col: int = row + 8 - (num - row - 1) // 10
    return row, col


# 赢得游戏的位置
WIN_LOC: Final[Tuple[int, int]] = (7, 0)

# 除了赢得游戏的条件之外，计算机总是会尝试移动到的位置
COMPUTER_SAFE_SPOTS: Final[FrozenSet[Tuple[int, int]]] = frozenset(
    [
        (2, 3),
        (4, 5),
        (5, 1),
        (6, 2),
    ]
)

# 计算机总是会尝试移动到的位置
COMPUTER_PREF_MOVES: Final[
    FrozenSet[Tuple[int, int]]
] = COMPUTER_SAFE_SPOTS | frozenset([WIN_LOC])

# 这些是位置（不包括赢得游戏的位置），任何玩家都可以强制赢得游戏（但计算机总是会选择其中一个COMPUTER_PREF_MOVES）。
SAFE_SPOTS: Final[FrozenSet[Tuple[int, int]]] = COMPUTER_SAFE_SPOTS | frozenset(
    [
        (0, 4),
        (3, 7),
    ]
)


def intro() -> None:
    """打印介绍，并在需要时打印说明。"""
    # 在屏幕上打印空格和"Queen"，总共33个空格
    print(" " * 33 + "Queen")
    # 在屏幕上打印空格和"Creative Computing  Morristown, New Jersey"，总共15个空格
    print(" " * 15 + "Creative Computing  Morristown, New Jersey")
    # 在屏幕上打印两个空行
    print("\n" * 2)
    # 如果用户选择了要求指令的话
    if ask("DO YOU WANT INSTRUCTIONS"):
        # 打印指令文本
        print(INSTR_TXT)
def get_move(current_loc: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    """Get the next move from the player."""
    prompt: str  # 用于存储提示信息的字符串变量
    player_resp: str  # 用于存储玩家输入的字符串变量
    move_raw: int  # 用于存储玩家输入的整数变量
    new_row: int  # 用于存储新的行坐标变量
    new_col: int  # 用于存储新的列坐标变量
    if current_loc is None:  # 如果当前位置为空（即第一轮）
        prompt = "WHERE WOULD YOU LIKE TO START? "  # 设置提示信息
    else:
        prompt = "WHAT IS YOUR MOVE? "  # 设置提示信息
        row, col = current_loc  # 获取当前位置的行列坐标
    while True:  # 进入循环，直到玩家输入合法的移动
        player_resp = input(prompt).strip()  # 获取玩家输入并去除首尾空格
        try:
            move_raw = int(player_resp)  # 尝试将玩家输入的字符串转换为整数
            if move_raw == 0:  # 如果玩家选择放弃
                return 8, 8  # 返回特定的行列坐标
            new_row, new_col = num_to_loc(move_raw)  # 将玩家输入的整数转换为新的行列坐标
            if current_loc is None:  # 如果当前位置为空（即第一轮）
                if (new_row == 0 or new_col == 7) and (
                    not FIX_BOARD_BUG or (new_col >= 0 and new_row < 8)
                ):
                    return new_row, new_col  # 返回新的行列坐标
                else:
                    prompt = (
                        "PLEASE READ THE DIRECTIONS AGAIN.\n"
                        "YOU HAVE BEGUN ILLEGALLY.\n\n"
                        "WHERE WOULD YOU LIKE TO START? "
                    )  # 设置提示信息
            else:
                if (
                    (new_row == row and new_col < col)  # 如果向左移动
                    or (new_col == col and new_row > row)  # 如果向下移动
                    or (new_row - row == col - new_col)  # 如果向左下对角线移动
                ) and (not FIX_BOARD_BUG or (new_col >= 0 and new_row < 8)):
                    return new_row, new_col  # 返回新的行列坐标
                else:
                    prompt = "Y O U   C H E A T . . .  TRY AGAIN? "  # 设置提示信息

        except ValueError:
            prompt = "!NUMBER EXPECTED - RETRY INPUT LINE\n? "  # 设置提示信息


def random_computer_move(location: Tuple[int, int]) -> Tuple[int, int]:
    """Make a random move."""
    row, col = location  # 获取计算机当前位置的行列坐标
    if (z := random()) > 0.6:  # 如果随机数大于0.6
        # Move down one space
        return row + 1, col  # 返回向下移动后的新的行列坐标
    elif z > 0.3:  # 如果随机数大于0.3
        # Move diagonaly (left and down) one space
        return row + 1, col - 1  # 返回向左下对角线移动后的新的行列坐标
    else:
        # 如果不满足上述条件，向左移动一个位置
        return row, col - 1
def computer_move(location: Tuple[int, int]) -> Tuple[int, int]:
    """Get the computer's move."""
    # 如果玩家已经做出了最佳移动，那么选择一个随机移动
    if location in SAFE_SPOTS:
        return random_computer_move(location)
    # 我们不需要实现检查玩家是否获胜的逻辑，因为在调用此函数之前已经检查过了。
    row, col = location
    for k in range(7, 0, -1):
        # 如果计算机可以向左移动 k 步并最终到达一个安全点或获胜，那么就这样做。
        if (new_loc := (row, col - k)) in COMPUTER_PREF_MOVES:
            return new_loc
        # 如果计算机可以向下移动 k 步并最终到达一个安全点或获胜，那么就这样做。
        if (new_loc := (row + k, col)) in COMPUTER_PREF_MOVES:
            return new_loc
        # 如果计算机可以斜向移动 k 步并最终到达一个安全点或获胜，那么就这样做。
        if (new_loc := (row + k, col - k)) in COMPUTER_PREF_MOVES:
            return new_loc
        # 作为备用，进行随机移动。（注意：这实际上不应该发生——如果玩家没有在 SAFE_SPOTS 中的位置上玩，应该总是能够做出最佳移动。）
    return random_computer_move(location)


def main_game() -> None:
    """Execute the main game."""
    game_over: bool = False
    location: Optional[Tuple[int, int]] = None  # 表示这是第一次移动
    # 当游戏未结束时执行循环
    while not game_over:
        # 获取玩家移动的位置
        location = get_move(location)
        # 如果玩家输入 (8, 8)，表示玩家放弃游戏
        if location == (8, 8):  
            # 打印玩家放弃游戏的消息
            print("\nIT LOOKS LIKE I HAVE WON BY FORFEIT.\n")
            # 设置游戏结束标志为 True
            game_over = True
        # 如果玩家移动到了胜利的位置
        elif location == WIN_LOC:  
            # 打印玩家胜利的消息
            print(WIN_MSG)
            # 设置游戏结束标志为 True
            game_over = True
        # 如果玩家未胜利，计算机进行移动
        else:
            location = computer_move(location)
            # 打印计算机移动到的位置
            print(f"COMPUTER MOVES TO SQUARE {loc_to_num(location)}")
            # 如果计算机移动到了胜利的位置
            if location == WIN_LOC:  
                # 打印计算机胜利的消息
                print(LOSE_MSG)
                # 设置游戏结束标志为 True
                game_over = True
        # 默认行为是不在每一轮显示游戏棋盘，但可以通过修改文件开头的标志来修改这一行为
        if not game_over and SHOW_BOARD_ALWAYS:
            # 如果游戏未结束且需要始终显示游戏棋盘，则打印游戏棋盘
            print(GAME_BOARD)
def ask(prompt: str) -> bool:
    """Ask a yes/no question until user gives an understandable response."""
    inpt: str
    while True:
        # Normalize input to uppercase, no whitespace, then get first character
        # 询问用户问题，并将用户输入的内容转换成大写，去除空格，然后取第一个字符
        inpt = input(prompt + "? ").upper().strip()[0]
        print()
        if inpt == "Y":
            return True
        elif inpt == "N":
            return False
        print("PLEASE ANSWER 'YES' OR 'NO'.")
    return False


if __name__ == "__main__":
    intro()
    still_playing: bool = True
    while still_playing:
        print(GAME_BOARD)
        main_game()
        still_playing = ask("ANYONE ELSE CARE TO TRY")
    print("\nOK --- THANKS AGAIN.")
```