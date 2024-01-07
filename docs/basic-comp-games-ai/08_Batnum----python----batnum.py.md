# `basic-computer-games\08_Batnum\python\batnum.py`

```

# 导入所需的模块
from enum import IntEnum
from typing import Any, Tuple

# 定义一个枚举类，表示游戏的胜利条件
class WinOptions(IntEnum):
    Undefined = 0
    TakeLast = 1
    AvoidLast = 2

    # 处理枚举值缺失的情况
    @classmethod
    def _missing_(cls, value: Any) -> "WinOptions":
        try:
            int_value = int(value)
        except Exception:
            return WinOptions.Undefined
        if int_value == 1:
            return WinOptions.TakeLast
        elif int_value == 2:
            return WinOptions.AvoidLast
        else:
            return WinOptions.Undefined

# 定义一个枚举类，表示游戏的起始条件
class StartOptions(IntEnum):
    Undefined = 0
    ComputerFirst = 1
    PlayerFirst = 2

    # 处理枚举值缺失的情况
    @classmethod
    def _missing_(cls, value: Any) -> "StartOptions":
        try:
            int_value = int(value)
        except Exception:
            return StartOptions.Undefined
        if int_value == 1:
            return StartOptions.ComputerFirst
        elif int_value == 2:
            return StartOptions.PlayerFirst
        else:
            return StartOptions.Undefined

# 打印游戏介绍和规则
def print_intro() -> None:
    """Print out the introduction and rules for the game."""
    # ... (打印游戏介绍和规则)

# 获取游戏参数
def get_params() -> Tuple[int, int, int, StartOptions, WinOptions]:
    """This requests the necessary parameters to play the game.

    Returns a set with the five game parameters:
        pile_size - the starting size of the object pile
        min_select - minimum selection that can be made on each turn
        max_select - maximum selection that can be made on each turn
        start_option - 1 if the computer is first
                      or 2 if the player is first
        win_option - 1 if the goal is to take the last object
                    or 2 if the goal is to not take the last object
    """
    # ... (获取游戏参数)

# 获取初始堆大小
def get_pile_size() -> int:
    # ... (获取初始堆大小)

# 获取胜利条件
def get_win_option() -> WinOptions:
    # ... (获取胜利条件)

# 获取最小和最大选择数量
def get_min_max() -> Tuple[int, int]:
    # ... (获取最小和最大选择数量)

# 获取起始条件
def get_start_option() -> StartOptions:
    # ... (获取起始条件)

# 玩家的回合
def player_move(
    pile_size: int, min_select: int, max_select: int, win_option: WinOptions
) -> Tuple[bool, int]:
    """This handles the player's turn - asking the player how many objects
    to take and doing some basic validation around that input.  Then it
    checks for any win conditions.

    Returns a boolean indicating whether the game is over and the new pile_size."""
    # ... (处理玩家的回合)

# 计算计算机选择的数量
def computer_pick(
    pile_size: int, min_select: int, max_select: int, win_option: WinOptions
) -> int:
    """This handles the logic to determine how many objects the computer
    will select on its turn.
    """
    # ... (计算计算机选择的数量)

# 计算机的回合
def computer_move(
    pile_size: int, min_select: int, max_select: int, win_option: WinOptions
) -> Tuple[bool, int]:
    """This handles the computer's turn - first checking for the various
    win/lose conditions and then calculating how many objects
    the computer will take.

    Returns a boolean indicating whether the game is over and the new pile_size."""
    # ... (处理计算机的回合)

# 游戏主循环
def play_game(
    pile_size: int,
    min_select: int,
    max_select: int,
    start_option: StartOptions,
    win_option: WinOptions,
) -> None:
    """This is the main game loop - repeating each turn until one
    of the win/lose conditions is met.
    """
    # ... (游戏主循环)

# 主函数
def main() -> None:
    # ... (主函数)

# 程序入口
if __name__ == "__main__":
    main()

```