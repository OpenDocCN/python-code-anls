# `d:/src/tocomm/basic-computer-games\08_Batnum\python\batnum.py`

```
from enum import IntEnum  # 导入 IntEnum 枚举类
from typing import Any, Tuple  # 导入 Any 和 Tuple 类型提示


class WinOptions(IntEnum):  # 定义名为 WinOptions 的枚举类，继承自 IntEnum
    Undefined = 0  # 枚举值 Undefined 的值为 0
    TakeLast = 1  # 枚举值 TakeLast 的值为 1
    AvoidLast = 2  # 枚举值 AvoidLast 的值为 2

    @classmethod  # 类方法装饰器，表示下面的方法是类方法
    def _missing_(cls, value: Any) -> "WinOptions":  # _missing_ 方法用于处理枚举值不存在的情况
        try:  # 尝试执行以下代码
            int_value = int(value)  # 将 value 转换为整数
        except Exception:  # 如果出现异常
            return WinOptions.Undefined  # 返回枚举值 Undefined
        if int_value == 1:  # 如果 int_value 的值为 1
            return WinOptions.TakeLast  # 返回枚举值 TakeLast
        elif int_value == 2:  # 如果 int_value 的值为 2
            return WinOptions.AvoidLast  # 返回枚举值 AvoidLast
        else:  # 如果 int_value 的值既不是 1 也不是 2
            return WinOptions.Undefined
```
这行代码返回WinOptions.Undefined。

```
class StartOptions(IntEnum):
    Undefined = 0
    ComputerFirst = 1
    PlayerFirst = 2
```
这段代码定义了一个枚举类StartOptions，其中包含了Undefined、ComputerFirst和PlayerFirst三个选项。

```
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
```
这段代码定义了一个类方法_missing_，用于处理枚举值不存在的情况。它尝试将传入的值转换为整数，如果转换失败则返回StartOptions.Undefined，如果转换成功则根据值返回对应的枚举选项。
def print_intro() -> None:
    """Print out the introduction and rules for the game."""
    # 打印游戏的介绍和规则
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
    # 获取初始堆大小
    pile_size = get_pile_size()
    # 如果堆大小小于0，则返回默认值
    if pile_size < 0:
        return (-1, 0, 0, StartOptions.Undefined, WinOptions.Undefined)
    # 获取胜利条件
    win_option = get_win_option()
    # 获取最小和最大选择数量
    min_select, max_select = get_min_max()
    # 获取起始选项
    start_option = get_start_option()
    return (pile_size, min_select, max_select, start_option, win_option)
    # 返回一个包含游戏参数的元组，包括堆大小、最小选择数、最大选择数、起始选项和获胜选项


def get_pile_size() -> int:
    # 获取堆大小的函数，返回一个整数
    pile_size = 0
    while pile_size == 0:
        try:
            pile_size = int(input("ENTER PILE SIZE "))
        except ValueError:
            pile_size = 0
    return pile_size


def get_win_option() -> WinOptions:
    # 获取获胜选项的函数，返回一个WinOptions枚举类型
    win_option: WinOptions = WinOptions.Undefined
    while win_option == WinOptions.Undefined:
        win_option = WinOptions(input("ENTER WIN OPTION - 1 TO TAKE LAST, 2 TO AVOID LAST: "))  # type: ignore
    return win_option
def get_min_max() -> Tuple[int, int]:
    # 初始化最小和最大选择为0
    min_select = 0
    max_select = 0
    # 循环直到输入的最小和最大选择都大于等于1且最小选择小于等于最大选择
    while min_select < 1 or max_select < 1 or min_select > max_select:
        # 从用户输入中获取最小和最大选择，并转换为整数
        (min_select, max_select) = (
            int(x) for x in input("ENTER MIN AND MAX ").split(" ")
        )
    # 返回最小和最大选择
    return min_select, max_select


def get_start_option() -> StartOptions:
    # 初始化开始选项为未定义
    start_option: StartOptions = StartOptions.Undefined
    # 循环直到用户输入有效的开始选项
    while start_option == StartOptions.Undefined:
        # 从用户输入中获取开始选项
        start_option = StartOptions(input("ENTER START OPTION - 1 COMPUTER FIRST, 2 YOU FIRST "))  # type: ignore
    # 返回开始选项
    return start_option


def player_move(
    pile_size: int, min_select: int, max_select: int, win_option: WinOptions
```
在这段代码中，我们定义了三个函数，分别是get_min_max，get_start_option和player_move。这些函数用于获取最小和最大选择、开始选项以及玩家的移动。每个函数都有特定的输入和输出，并且通过注释对代码进行了解释。
) -> Tuple[bool, int]:  # 定义函数的返回类型为布尔值和整数元组
    """This handles the player's turn - asking the player how many objects
    to take and doing some basic validation around that input.  Then it
    checks for any win conditions.

    Returns a boolean indicating whether the game is over and the new pile_size."""  # 函数的文档字符串，解释了函数的作用和返回值
    player_done = False  # 初始化玩家是否完成回合的标志为False
    while not player_done:  # 当玩家未完成回合时循环
        player_move = int(input("YOUR MOVE "))  # 获取玩家输入的移动数量
        if player_move == 0:  # 如果玩家输入为0
            print("I TOLD YOU NOT TO USE ZERO!  COMPUTER WINS BY FORFEIT.")  # 打印消息，电脑因放弃而获胜
            return (True, pile_size)  # 返回游戏结束的标志和当前堆的大小
        if player_move > max_select or player_move < min_select:  # 如果玩家输入大于最大可选数量或小于最小可选数量
            print("ILLEGAL MOVE, REENTER IT")  # 打印消息，要求重新输入
            continue  # 继续循环
        pile_size = pile_size - player_move  # 更新堆的大小
        player_done = True  # 玩家完成回合
        if pile_size <= 0:  # 如果堆的大小小于等于0
            if win_option == WinOptions.AvoidLast:  # 如果胜利条件为避免最后一次取数
                print("TOUGH LUCK, YOU LOSE.")  # 打印消息，玩家输了
    else:
        # 如果条件不满足，则打印"CONGRATULATIONS, YOU WIN."
        print("CONGRATULATIONS, YOU WIN.")
    # 返回一个元组，第一个元素为True，第二个元素为pile_size
    return (True, pile_size)
# 如果条件不满足，则返回一个元组，第一个元素为False，第二个元素为pile_size
return (False, pile_size)

def computer_pick(
    pile_size: int, min_select: int, max_select: int, win_option: WinOptions
) -> int:
    """This handles the logic to determine how many objects the computer
    will select on its turn.
    """
    # 根据游戏规则和当前堆的大小，计算电脑选择的数量
    q = pile_size - 1 if win_option == WinOptions.AvoidLast else pile_size
    c = min_select + max_select
    computer_pick = q - (c * int(q / c))
    # 如果计算出的数量小于最小选择数量，则选择最小选择数量
    if computer_pick < min_select:
        computer_pick = min_select
    # 如果计算出的数量大于最大选择数量，则选择最大选择数量
    if computer_pick > max_select:
        computer_pick = max_select
    # 返回电脑选择的数量
    return computer_pick
# 定义一个名为computer_move的函数，接受四个参数：pile_size（堆的大小）、min_select（最小选择数量）、max_select（最大选择数量）、win_option（胜利选项）
# 返回一个元组，包含一个布尔值表示游戏是否结束，以及新的堆大小
def computer_move(
    pile_size: int, min_select: int, max_select: int, win_option: WinOptions
) -> Tuple[bool, int]:
    """This handles the computer's turn - first checking for the various
    win/lose conditions and then calculating how many objects
    the computer will take.

    Returns a boolean indicating whether the game is over and the new pile_size."""
    # 首先，检查此次移动的胜利条件
    # 在这种情况下，我们通过拿走最后一个对象并且
    # 剩下的堆小于最大选择数量
    # 所以计算机可以拿走它们并赢得比赛
    if win_option == WinOptions.TakeLast and pile_size <= max_select:
        print(f"COMPUTER TAKES {pile_size} AND WINS.")
        return (True, pile_size)
    # 在这种情况下，我们通过拿走最后一个对象而输掉比赛，并且
    # 剩下的堆小于最小选择数量，计算机必须拿走它们全部。
    if win_option == WinOptions.AvoidLast and pile_size <= min_select:
        # 如果选择避免最后一颗石头，并且当前石头数量小于等于最小可选数量，计算机选择最小可选数量并且输掉游戏
        print(f"COMPUTER TAKES {min_select} AND LOSES.")
        return (True, pile_size)

    # 否则，确定计算机选择的数量
    curr_sel = computer_pick(pile_size, min_select, max_select, win_option)
    # 更新石头数量
    pile_size = pile_size - curr_sel
    # 打印计算机选择的数量和剩余石头数量
    print(f"COMPUTER TAKES {curr_sel} AND LEAVES {pile_size}")
    return (False, pile_size)
    """
    # 初始化游戏结束标志为 False
    game_over = False
    # players_turn 是一个布尔值，用于跟踪是玩家还是计算机的回合
    players_turn = start_option == StartOptions.PlayerFirst

    # 当游戏未结束时循环执行以下操作
    while not game_over:
        # 如果是玩家的回合
        if players_turn:
            # 调用 player_move 函数执行玩家的移动，并更新游戏结束标志和堆大小
            (game_over, pile_size) = player_move(
                pile_size, min_select, max_select, win_option
            )
            # 切换到计算机的回合
            players_turn = False
            # 如果游戏结束，则返回
            if game_over:
                return
        # 如果不是玩家的回合
        if not players_turn:
            # 调用 computer_move 函数执行计算机的移动，并更新游戏结束标志和堆大小
            (game_over, pile_size) = computer_move(
                pile_size, min_select, max_select, win_option
            )
            # 切换到玩家的回合
            players_turn = True
    """
# 定义主函数
def main() -> None:
    # 无限循环，直到用户通过 ctrl-C 终止程序
    while True:
        # 打印游戏介绍
        print_intro()
        # 获取游戏参数：堆大小、最小选择数、最大选择数、起始选项、获胜选项
        (pile_size, min_select, max_select, start_option, win_option) = get_params()

        # 如果堆大小小于0，结束程序
        if pile_size < 0:
            return

        # 继续进行游戏，直到用户通过 ctrl-C 终止程序
        play_game(pile_size, min_select, max_select, start_option, win_option)


# 如果当前脚本为主程序，则执行主函数
if __name__ == "__main__":
    main()
```