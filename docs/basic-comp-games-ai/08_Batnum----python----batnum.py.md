# `basic-computer-games\08_Batnum\python\batnum.py`

```
# 导入需要的模块
from enum import IntEnum
from typing import Any, Tuple

# 定义一个枚举类，表示游戏中的胜利选项
class WinOptions(IntEnum):
    Undefined = 0
    TakeLast = 1
    AvoidLast = 2

    # 定义一个方法，用于处理枚举值缺失的情况
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

# 定义一个枚举类，表示游戏中的起始选项
class StartOptions(IntEnum):
    Undefined = 0
    ComputerFirst = 1
    PlayerFirst = 2

    # 定义一个方法，用于处理枚举值缺失的情况
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

# 定义一个函数，用于打印游戏介绍和规则
def print_intro() -> None:
    """Print out the introduction and rules for the game."""
    print("BATNUM".rjust(33, " "))
    print("CREATIVE COMPUTING  MORRISSTOWN, NEW JERSEY".rjust(15, " "))
    print()
    print()
    print()
    print("THIS PROGRAM IS A 'BATTLE OF NUMBERS' GAME, WHERE THE")
    print("COMPUTER IS YOUR OPPONENT.")
    print()
    print("THE GAME STARTS WITH AN ASSUMED PILE OF OBJECTS. YOU")
    print("AND YOUR OPPONENT ALTERNATELY REMOVE OBJECTS FROM THE PILE.")
    print("WINNING IS DEFINED IN ADVANCE AS TAKING THE LAST OBJECT OR")
    print("NOT. YOU CAN ALSO SPECIFY SOME OTHER BEGINNING CONDITIONS.")
    print("DON'T USE ZERO, HOWEVER, IN PLAYING THE GAME.")
    print("ENTER A NEGATIVE NUMBER FOR NEW PILE SIZE TO STOP PLAYING.")
    print()
    return

# 定义一个函数，用于获取游戏参数
def get_params() -> Tuple[int, int, int, StartOptions, WinOptions]:
    """This requests the necessary parameters to play the game.
    # 返回一个包含五个游戏参数的集合：
    # pile_size - 物体堆的初始大小
    # min_select - 每轮可以选择的最小数量
    # max_select - 每轮可以选择的最大数量
    # start_option - 如果计算机先手则为1，如果玩家先手则为2
    # win_option - 如果目标是取走最后一个物体则为1，如果目标是不取走最后一个物体则为2
    """
    # 获取物体堆的大小
    pile_size = get_pile_size()
    # 如果物体堆的大小小于0，则返回默认值
    if pile_size < 0:
        return (-1, 0, 0, StartOptions.Undefined, WinOptions.Undefined)
    # 获取赢得游戏的选项
    win_option = get_win_option()
    # 获取每轮可以选择的最小和最大数量
    min_select, max_select = get_min_max()
    # 获取先手选项
    start_option = get_start_option()
    # 返回包含游戏参数的元组
    return (pile_size, min_select, max_select, start_option, win_option)
# 获取堆大小，返回一个整数
def get_pile_size() -> int:
    # 如果堆大小为负数，游戏结束
    pile_size = 0
    while pile_size == 0:
        try:
            pile_size = int(input("ENTER PILE SIZE "))
        except ValueError:
            pile_size = 0
    return pile_size


# 获取获胜选项，返回WinOptions枚举类型
def get_win_option() -> WinOptions:
    win_option: WinOptions = WinOptions.Undefined
    while win_option == WinOptions.Undefined:
        win_option = WinOptions(input("ENTER WIN OPTION - 1 TO TAKE LAST, 2 TO AVOID LAST: "))  # type: ignore
    return win_option


# 获取最小和最大选择范围，返回一个元组
def get_min_max() -> Tuple[int, int]:
    min_select = 0
    max_select = 0
    while min_select < 1 or max_select < 1 or min_select > max_select:
        (min_select, max_select) = (
            int(x) for x in input("ENTER MIN AND MAX ").split(" ")
        )
    return min_select, max_select


# 获取起始选项，返回StartOptions枚举类型
def get_start_option() -> StartOptions:
    start_option: StartOptions = StartOptions.Undefined
    while start_option == StartOptions.Undefined:
        start_option = StartOptions(input("ENTER START OPTION - 1 COMPUTER FIRST, 2 YOU FIRST "))  # type: ignore
    return start_option


# 处理玩家的移动，返回一个布尔值和新的堆大小
def player_move(
    pile_size: int, min_select: int, max_select: int, win_option: WinOptions
) -> Tuple[bool, int]:
    """This handles the player's turn - asking the player how many objects
    to take and doing some basic validation around that input.  Then it
    checks for any win conditions.

    Returns a boolean indicating whether the game is over and the new pile_size."""
    player_done = False
    # 当玩家未完成时执行循环
    while not player_done:
        # 获取玩家输入的移动步数
        player_move = int(input("YOUR MOVE "))
        # 如果玩家输入0，则打印消息并返回（True，pile_size）
        if player_move == 0:
            print("I TOLD YOU NOT TO USE ZERO!  COMPUTER WINS BY FORFEIT.")
            return (True, pile_size)
        # 如果玩家输入的移动步数大于最大可选或小于最小可选，则打印消息并继续循环
        if player_move > max_select or player_move < min_select:
            print("ILLEGAL MOVE, REENTER IT")
            continue
        # 更新堆大小，减去玩家的移动步数
        pile_size = pile_size - player_move
        # 设置玩家完成标志为True
        player_done = True
        # 如果堆大小小于等于0
        if pile_size <= 0:
            # 如果胜利选项为避免最后一步，则打印消息“TOUGH LUCK, YOU LOSE.”
            if win_option == WinOptions.AvoidLast:
                print("TOUGH LUCK, YOU LOSE.")
            # 否则打印消息“CONGRATULATIONS, YOU WIN.”
            else:
                print("CONGRATULATIONS, YOU WIN.")
            # 返回（True，pile_size）
            return (True, pile_size)
    # 返回（False，pile_size）
    return (False, pile_size)
# 定义一个函数，用于确定计算机在轮到它的时候选择多少个对象
def computer_pick(
    pile_size: int, min_select: int, max_select: int, win_option: WinOptions
) -> int:
    """This handles the logic to determine how many objects the computer
    will select on its turn.
    """
    # 根据游戏规则和当前堆的大小计算一个值
    q = pile_size - 1 if win_option == WinOptions.AvoidLast else pile_size
    # 计算最小和最大选择数的和
    c = min_select + max_select
    # 计算计算机选择的对象数量
    computer_pick = q - (c * int(q / c))
    # 如果计算机选择的对象数量小于最小选择数，则选择最小选择数
    if computer_pick < min_select:
        computer_pick = min_select
    # 如果计算机选择的对象数量大于最大选择数，则选择最大选择数
    if computer_pick > max_select:
        computer_pick = max_select
    # 返回计算机选择的对象数量
    return computer_pick


def computer_move(
    pile_size: int, min_select: int, max_select: int, win_option: WinOptions
) -> Tuple[bool, int]:
    """This handles the computer's turn - first checking for the various
    win/lose conditions and then calculating how many objects
    the computer will take.

    Returns a boolean indicating whether the game is over and the new pile_size."""
    # 首先，检查此次移动的胜利条件
    # 在这种情况下，我们通过拿走最后一个对象并且剩余堆的大小小于最大选择数来获胜
    # 因此计算机可以拿走它们所有并获胜
    if win_option == WinOptions.TakeLast and pile_size <= max_select:
        print(f"COMPUTER TAKES {pile_size} AND WINS.")
        return (True, pile_size)
    # 在这种情况下，我们通过拿走最后一个对象并且剩余堆的大小小于最小选择数来失败
    # 计算机必须拿走它们所有
    if win_option == WinOptions.AvoidLast and pile_size <= min_select:
        print(f"COMPUTER TAKES {min_select} AND LOSES.")
        return (True, pile_size)

    # 否则，我们确定计算机选择的对象数量
    curr_sel = computer_pick(pile_size, min_select, max_select, win_option)
    pile_size = pile_size - curr_sel
    print(f"COMPUTER TAKES {curr_sel} AND LEAVES {pile_size}")
    return (False, pile_size)


def play_game(
    pile_size: int,
    min_select: int,
    max_select: int,
    start_option: StartOptions,
    win_option: WinOptions,
) -> None:
    """这是主游戏循环 - 每个回合重复，直到满足胜利/失败条件之一。"""
    # game_over 是一个布尔值，用于跟踪游戏是否结束
    game_over = False
    # players_turn 是一个布尔值，用于跟踪是玩家还是计算机的回合
    players_turn = start_option == StartOptions.PlayerFirst

    while not game_over:
        if players_turn:
            # 调用玩家移动函数，更新游戏状态并返回游戏是否结束以及堆大小
            (game_over, pile_size) = player_move(
                pile_size, min_select, max_select, win_option
            )
            players_turn = False
            # 如果游戏结束，直接返回
            if game_over:
                return
        if not players_turn:
            # 调用计算机移动函数，更新游戏状态并返回游戏是否结束以及堆大小
            (game_over, pile_size) = computer_move(
                pile_size, min_select, max_select, win_option
            )
            players_turn = True
# 定义主函数，没有返回值
def main() -> None:
    # 无限循环，直到用户手动退出
    while True:
        # 打印游戏介绍
        print_intro()
        # 获取游戏参数：堆大小、最小选择数、最大选择数、起始选项、获胜选项
        (pile_size, min_select, max_select, start_option, win_option) = get_params()

        # 如果堆大小小于0，退出程序
        if pile_size < 0:
            return

        # 持续进行游戏，直到用户使用 ctrl-C 终止
        play_game(pile_size, min_select, max_select, start_option, win_option)


# 如果当前脚本为主程序，则执行主函数
if __name__ == "__main__":
    main()
```