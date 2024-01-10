# `basic-computer-games\30_Cube\python\cube.py`

```
#!/usr/bin/env python3
# 指定脚本解释器为 Python 3

"""
CUBE

Converted from BASIC to Python by Trevor Hobson
"""
# 多行注释，描述游戏的名称和作者

import random
from typing import Tuple
# 导入 random 模块和 Tuple 类型提示

def mine_position() -> Tuple[int, int, int]:
    # 返回一个包含三个随机整数的元组
    return (random.randint(1, 3), random.randint(1, 3), random.randint(1, 3))

def parse_move(move: str) -> Tuple[int, int, int]:
    # 将输入的字符串解析为包含三个整数的元组
    coordinates = [int(item) for item in move.split(",")]
    if len(coordinates) == 3:
        return tuple(coordinates)  # type: ignore
    raise ValueError

def play_game() -> None:
    """Play one round of the game"""
    # 初始化玩家的初始资金
    money = 500
    print("\nYou have", money, "dollars.")
    print("\nTough luck")
    print("\nGoodbye.")

def print_instructions() -> None:
    # 打印游戏的玩法说明
    print("\nThis is a game in which you will be playing against the")
    print("random decisions of the computer. The field of play is a")
    print("cube of side 3. Any of the 27 locations can be designated")
    print("by inputing three numbers such as 2,3,1. At the start,")
    print("you are automatically at location 1,1,1. The object of")
    print("the game is to get to location 3,3,3. One minor detail:")
    print("the computer will pick, at random, 5 locations at which")
    print("it will plant land mines. If you hit one of these locations")
    print("you lose. One other detail: You may move only one space")
    print("in one direction each move. For example: From 1,1,2 you")
    print("may move to 2,1,2 or 1,1,3. You may not change")
    print("two of the numbers on the same move. If you make an illegal")
    print("move, you lose and the computer takes the money you may")
    print("have bet on that round.\n")
    print("When stating the amount of a wager, print only the number")
    print("of dollars (example: 250) you are automatically started with")
    print("500 dollars in your account.\n")
    print("Good luck!")

def main() -> None:
    # 打印游戏名称和作者信息
    print(" " * 34 + "CUBE")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    # 如果用户输入以字母 "y" 开头的字符串，表示用户想要查看游戏说明，调用打印游戏说明的函数
    if input("Do you want to see the instructions ").lower().startswith("y"):
        print_instructions()

    # 设置一个变量，用于控制是否继续游戏的循环
    keep_playing = True
    # 当 keep_playing 为 True 时，持续进行游戏
    while keep_playing:
        # 调用游戏函数进行游戏
        play_game()
        # 用户输入是否继续游戏，如果以字母 "y" 开头的字符串，继续游戏，否则结束游戏
        keep_playing = input("\nPlay again? (yes or no) ").lower().startswith("y")
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```