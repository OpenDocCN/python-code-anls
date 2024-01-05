# `d:/src/tocomm/basic-computer-games\35_Even_Wins\python\evenwins.py`

```
# 导入 dataclasses 模块中的 dataclass 类
from dataclasses import dataclass
from typing import Literal, Tuple  # 导入类型提示模块，用于声明变量的类型

PlayerType = Literal["human", "computer"]  # 声明一个类型别名，表示玩家类型为人类或计算机

@dataclass  # 使用装饰器声明一个数据类
class MarbleCounts:
    middle: int  # 中间的弹珠数量
    human: int  # 人类玩家的弹珠数量
    computer: int  # 计算机玩家的弹珠数量

def print_intro() -> None:  # 定义一个不返回任何值的函数
    print("Welcome to Even Wins!")  # 打印欢迎消息
    print("Based on evenwins.bas from Creative Computing")  # 打印基于Creative Computing的evenwins.bas
    print()  # 打印空行
    print("Even Wins is a two-person game. You start with")  # 打印游戏介绍
    print("27 marbles in the middle of the table.")  # 打印游戏介绍
    print()  # 打印空行
    print("Players alternate taking marbles from the middle.")  # 打印游戏介绍
    # 打印游戏规则说明
    print("A player can take 1 to 4 marbles on their turn, and")
    print("turns cannot be skipped. The game ends when there are")
    print("no marbles left, and the winner is the one with an even")
    print("number of marbles.")
    print()

# 根据给定的数量返回对应的字符串描述
def marbles_str(n: int) -> str:
    # 如果数量为1，则返回"1 marble"
    if n == 1:
        return "1 marble"
    # 否则返回对应数量的字符串描述
    return f"{n} marbles"

# 选择先手玩家
def choose_first_player() -> PlayerType:
    # 循环直到得到有效的输入
    while True:
        # 获取用户输入
        ans = input("Do you want to play first? (y/n) --> ")
        # 如果用户选择是，则返回"human"
        if ans == "y":
            return "human"
        # 如果用户选择否，则返回"computer"
        elif ans == "n":
            return "computer"
        else:
            # 打印提示信息，让用户选择先手还是后手
            print()
            print('Please enter "y" if you want to play first,')
            print('or "n" if you want to play second.')
            print()


def toggle_player(whose_turn: PlayerType) -> PlayerType:
    # 切换玩家角色，如果当前是人类玩家，则切换为电脑玩家，反之亦然
    if whose_turn == "human":
        return "computer"
    else:
        return "human"


def to_int(s: str) -> Tuple[bool, int]:
    """Convert a string s to an int, if possible."""
    # 尝试将字符串转换为整数，如果成功则返回True和转换后的整数，否则返回False和None
    try:
        n = int(s)
        return True, n
    except Exception:
        return False, 0
```
这行代码是一个函数的返回语句，返回一个布尔值False和整数0。

```
def print_board(marbles: MarbleCounts) -> None:
```
这行代码定义了一个名为print_board的函数，它接受一个名为marbles的参数，该参数的类型为MarbleCounts。函数的返回类型为None。

```
    print()
    print(f" marbles in the middle: {marbles.middle} " + marbles.middle * "*")
    print(f"    # marbles you have: {marbles.human}")
    print(f"# marbles computer has: {marbles.computer}")
    print()
```
这几行代码用于打印游戏板的状态，包括中间的弹珠数量，玩家拥有的弹珠数量和计算机拥有的弹珠数量。

```
def human_turn(marbles: MarbleCounts) -> None:
```
这行代码定义了一个名为human_turn的函数，它接受一个名为marbles的参数，该参数的类型为MarbleCounts。函数的返回类型为None。

```
    """get number in range 1 to min(4, marbles.middle)"""
```
这行代码是一个文档字符串，用于描述函数的作用，即获取一个介于1和min(4, marbles.middle)之间的数字。

```
    max_choice = min(4, marbles.middle)
    print("It's your turn!")
    while True:
        s = input(f"Marbles to take? (1 - {max_choice}) --> ")
        ok, n = to_int(s)
        if not ok:
            print(f"\n  Please enter a whole number from 1 to {max_choice}\n")
```
这几行代码用于提示玩家进行操作，获取玩家输入的数字，并进行验证。如果输入不符合要求，则提示玩家重新输入。
        continue  # 继续循环，跳过当前迭代的剩余部分
        if n < 1:  # 如果 n 小于 1
            print("\n  You must take at least 1 marble!\n")  # 打印信息，至少需要拿一个弹珠
            continue  # 继续循环，跳过当前迭代的剩余部分
        if n > max_choice:  # 如果 n 大于最大可选数量
            print(f"\n  You can take at most {marbles_str(max_choice)}\n")  # 打印信息，最多可以拿的数量
            continue  # 继续循环，跳过当前迭代的剩余部分
        print(f"\nOkay, taking {marbles_str(n)} ...")  # 打印信息，确认拿取的数量
        marbles.middle -= n  # 更新中间弹珠数量
        marbles.human += n  # 更新玩家拥有的弹珠数量
        return  # 返回

def game_over(marbles: MarbleCounts) -> None:
    print()  # 打印空行
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  # 打印游戏结束提示
    print("!! All the marbles are taken: Game Over!")  # 打印游戏结束提示
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  # 打印游戏结束提示
    print()  # 打印空行
    print_board(marbles)  # 调用打印游戏板的函数
    # 如果玩家的数量是偶数，打印玩家获胜的消息
    if marbles.human % 2 == 0:
        print("You are the winner! Congratulations!")
    # 如果玩家的数量是奇数，打印计算机获胜的消息
    else:
        print("The computer wins: all hail mighty silicon!")
    # 打印空行
    print()


def computer_turn(marbles: MarbleCounts) -> None:
    # 初始化计算机要取的弹珠数量
    marbles_to_take = 0

    # 打印提示信息，表示轮到计算机了
    print("It's the computer's turn ...")
    # 计算中间弹珠数量除以6的余数
    r = marbles.middle - 6 * int(marbles.middle / 6)

    # 如果玩家弹珠数量除以2的商等于玩家弹珠数量除以2
    if int(marbles.human / 2) == marbles.human / 2:
        # 如果余数小于1.5或大于5.3，计算机取1个弹珠
        if r < 1.5 or r > 5.3:
            marbles_to_take = 1
        # 否则，计算机取余数减1个弹珠
        else:
            marbles_to_take = r - 1

    # 如果中间弹珠数量小于4.2
    elif marbles.middle < 4.2:
        marbles_to_take = marbles.middle  # 从中间取出的弹珠数量等于中间弹珠的数量
    elif r > 3.4:  # 如果r大于3.4
        if r < 4.7 or r > 3.5:  # 如果r小于4.7或者大于3.5
            marbles_to_take = 4  # 取出4个弹珠
    else:  # 否则
        marbles_to_take = r + 1  # 取出r加1个弹珠

    print(f"Computer takes {marbles_str(marbles_to_take)} ...")  # 打印电脑取出的弹珠数量
    marbles.middle -= marbles_to_take  # 从中间弹珠数量中减去电脑取出的弹珠数量
    marbles.computer += marbles_to_take  # 电脑的弹珠数量增加取出的弹珠数量


def play_game(whose_turn: PlayerType) -> None:  # 定义一个玩游戏的函数，参数为轮到谁的类型，返回值为None
    marbles = MarbleCounts(middle=27, human=0, computer=0)  # 初始化弹珠数量，中间27个，人类和电脑都为0
    print_board(marbles)  # 打印当前的弹珠情况

    while True:  # 无限循环
        if marbles.middle == 0:  # 如果中间弹珠数量为0
            game_over(marbles)  # 游戏结束
            return  # 返回
        elif whose_turn == "human":  # 如果轮到玩家下棋
            human_turn(marbles)  # 玩家进行下棋操作
            print_board(marbles)  # 打印当前棋盘状态
            whose_turn = toggle_player(whose_turn)  # 切换到电脑下棋
        elif whose_turn == "computer":  # 如果轮到电脑下棋
            computer_turn(marbles)  # 电脑进行下棋操作
            print_board(marbles)  # 打印当前棋盘状态
            whose_turn = toggle_player(whose_turn)  # 切换到玩家下棋
        else:  # 如果轮到的玩家既不是玩家也不是电脑
            raise Exception(f"whose_turn={whose_turn} is not 'human' or 'computer'")  # 抛出异常，提示轮到的玩家不是玩家也不是电脑


def main() -> None:
    print_intro()  # 打印游戏介绍

    while True:  # 无限循环，直到游戏结束
        whose_turn = choose_first_player()  # 选择先手玩家
        play_game(whose_turn)  # 开始游戏，传入先手玩家

        print()  # 打印空行，用于分隔不同游戏的输出
        again = input("Would you like to play again? (y/n) --> ").lower()  # 询问用户是否想再玩一次游戏，并将用户输入转换为小写字母
        if again == "y":  # 如果用户输入是"y"
            print("\nOk, let's play again ...\n")  # 打印消息表示再次开始游戏
        else:  # 如果用户输入不是"y"
            print("\nOk, thanks for playing ... goodbye!\n")  # 打印消息表示感谢用户玩游戏并结束游戏
            return  # 返回到调用该函数的地方，结束程序的执行


if __name__ == "__main__":  # 如果当前文件被直接运行而不是被导入
    main()  # 调用主函数进行程序的执行
```