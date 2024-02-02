# `basic-computer-games\35_Even_Wins\python\evenwins.py`

```py
"""
This version of evenwins.bas based on game decscription and does *not*
follow the source. The computer chooses marbles at random.

For simplicity, global variables are used to store the game state.
A good exercise would be to replace this with a class.
The code is not short, but hopefully it is easy for beginners to understand
and modify.

Infinite loops of the style "while True:" are used to simplify some of the
code. The "continue" keyword is used in a few places to jump back to the top
of the loop. The "return" keyword is also used to break out of functions.
This is generally considered poor style, but in this case it simplifies the
code and makes it easier to read (at least in my opinion). A good exercise
would be to remove these infinite loops, and uses of continue, to follow a
more structured style.
"""

# 导入必要的模块
from dataclasses import dataclass
from typing import Literal, Tuple

# 定义类型别名
PlayerType = Literal["human", "computer"]

# 定义数据类，用于存储各种类型的弹珠数量
@dataclass
class MarbleCounts:
    middle: int
    human: int
    computer: int

# 打印游戏介绍
def print_intro() -> None:
    print("Welcome to Even Wins!")
    print("Based on evenwins.bas from Creative Computing")
    print()
    print("Even Wins is a two-person game. You start with")
    print("27 marbles in the middle of the table.")
    print()
    print("Players alternate taking marbles from the middle.")
    print("A player can take 1 to 4 marbles on their turn, and")
    print("turns cannot be skipped. The game ends when there are")
    print("no marbles left, and the winner is the one with an even")
    print("number of marbles.")
    print()

# 根据数量返回对应的字符串描述
def marbles_str(n: int) -> str:
    if n == 1:
        return "1 marble"
    return f"{n} marbles"

# 选择先手玩家
def choose_first_player() -> PlayerType:
    # 无限循环，直到条件被满足才会退出
    while True:
        # 获取用户输入，询问是否要先行
        ans = input("Do you want to play first? (y/n) --> ")
        # 如果用户输入是"y"，返回"human"
        if ans == "y":
            return "human"
        # 如果用户输入是"n"，返回"computer"
        elif ans == "n":
            return "computer"
        # 如果用户输入既不是"y"也不是"n"，则提示用户重新输入
        else:
            print()
            print('Please enter "y" if you want to play first,')
            print('or "n" if you want to play second.')
            print()
# 切换玩家，如果当前是人类玩家，则返回计算机玩家，否则返回人类玩家
def toggle_player(whose_turn: PlayerType) -> PlayerType:
    if whose_turn == "human":
        return "computer"
    else:
        return "human"


# 将字符串 s 转换为整数，如果可能的话
def to_int(s: str) -> Tuple[bool, int]:
    try:
        n = int(s)
        return True, n
    except Exception:
        return False, 0


# 打印游戏棋盘上的信息
def print_board(marbles: MarbleCounts) -> None:
    print()
    print(f" marbles in the middle: {marbles.middle} " + marbles.middle * "*")
    print(f"    # marbles you have: {marbles.human}")
    print(f"# marbles computer has: {marbles.computer}")
    print()


# 人类玩家的回合
def human_turn(marbles: MarbleCounts) -> None:
    """get number in range 1 to min(4, marbles.middle)"""
    max_choice = min(4, marbles.middle)
    print("It's your turn!")
    while True:
        s = input(f"Marbles to take? (1 - {max_choice}) --> ")
        ok, n = to_int(s)
        if not ok:
            print(f"\n  Please enter a whole number from 1 to {max_choice}\n")
            continue
        if n < 1:
            print("\n  You must take at least 1 marble!\n")
            continue
        if n > max_choice:
            print(f"\n  You can take at most {marbles_str(max_choice)}\n")
            continue
        print(f"\nOkay, taking {marbles_str(n)} ...")
        marbles.middle -= n
        marbles.human += n
        return


# 游戏结束
def game_over(marbles: MarbleCounts) -> None:
    print()
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!! All the marbles are taken: Game Over!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print()
    print_board(marbles)
    if marbles.human % 2 == 0:
        print("You are the winner! Congratulations!")
    else:
        print("The computer wins: all hail mighty silicon!")
    print()


# 计算机玩家的回合
def computer_turn(marbles: MarbleCounts) -> None:
    marbles_to_take = 0

    print("It's the computer's turn ...")
    r = marbles.middle - 6 * int(marbles.middle / 6)
    # 如果人类拿的弹珠数除以2等于整数，进入条件判断
    if int(marbles.human / 2) == marbles.human / 2:
        # 如果r小于1.5或者大于5.3，设置电脑拿的弹珠数为1，否则设置为r-1
        if r < 1.5 or r > 5.3:
            marbles_to_take = 1
        else:
            marbles_to_take = r - 1

    # 如果人类拿的弹珠数除以2不等于整数，进入下一个条件判断
    elif marbles.middle < 4.2:
        # 设置电脑拿的弹珠数为中间位置的弹珠数
        marbles_to_take = marbles.middle
    # 如果中间位置的弹珠数大于等于4.2，进入下一个条件判断
    elif r > 3.4:
        # 如果r大于3.4且小于4.7或者大于3.5，设置电脑拿的弹珠数为4
        if r < 4.7 or r > 3.5:
            marbles_to_take = 4
    # 如果以上条件都不满足，设置电脑拿的弹珠数为r+1
    else:
        marbles_to_take = r + 1

    # 打印电脑拿的弹珠数
    print(f"Computer takes {marbles_str(marbles_to_take)} ...")
    # 更新中间位置的弹珠数
    marbles.middle -= marbles_to_take
    # 更新电脑拿的弹珠数
    marbles.computer += marbles_to_take
# 定义一个函数，用于进行游戏，参数为当前轮到的玩家类型，无返回值
def play_game(whose_turn: PlayerType) -> None:
    # 初始化游戏的弹珠数量，中间位置27个，玩家和电脑都为0个
    marbles = MarbleCounts(middle=27, human=0, computer=0)
    # 打印游戏板的状态
    print_board(marbles)

    # 进入游戏循环
    while True:
        # 如果中间位置的弹珠数量为0，游戏结束
        if marbles.middle == 0:
            game_over(marbles)
            return
        # 如果轮到玩家
        elif whose_turn == "human":
            # 玩家进行操作
            human_turn(marbles)
            # 打印游戏板的状态
            print_board(marbles)
            # 切换到电脑的回合
            whose_turn = toggle_player(whose_turn)
        # 如果轮到电脑
        elif whose_turn == "computer":
            # 电脑进行操作
            computer_turn(marbles)
            # 打印游戏板的状态
            print_board(marbles)
            # 切换到玩家的回合
            whose_turn = toggle_player(whose_turn)
        # 如果轮到的玩家类型不是"human"或"computer"，抛出异常
        else:
            raise Exception(f"whose_turn={whose_turn} is not 'human' or 'computer'")


# 定义一个函数，用于主程序逻辑，无返回值
def main() -> None:
    # 打印游戏介绍
    print_intro()

    # 进入游戏循环
    while True:
        # 选择先手玩家
        whose_turn = choose_first_player()
        # 开始游戏
        play_game(whose_turn)

        # 打印空行
        print()
        # 询问是否再玩一局
        again = input("Would you like to play again? (y/n) --> ").lower()
        # 如果回答是"y"，继续下一局游戏
        if again == "y":
            print("\nOk, let's play again ...\n")
        # 如果回答不是"y"，结束游戏
        else:
            print("\nOk, thanks for playing ... goodbye!\n")
            return


# 如果当前脚本为主程序，则执行main函数
if __name__ == "__main__":
    main()
```