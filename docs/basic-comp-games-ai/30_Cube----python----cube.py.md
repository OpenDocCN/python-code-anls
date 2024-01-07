# `basic-computer-games\30_Cube\python\cube.py`

```

#!/usr/bin/env python3
# 指定使用 Python3 解释器

"""
CUBE

Converted from BASIC to Python by Trevor Hobson
"""
# 程序的简要介绍

import random
from typing import Tuple
# 导入 random 模块和 Tuple 类型

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
    # 玩游戏的函数

    money = 500
    print("\nYou have", money, "dollars.")
    # 打印玩家初始金额
    while True:
        mines = []
        for _ in range(5):
            while True:
                mine = mine_position()
                if not (mine in mines or mine == (1, 1, 1) or mine == (3, 3, 3)):
                    break
            mines.append(mine)
        # 生成5个地雷的位置
        wager = -1
        while wager == -1:
            try:
                wager = int(input("\nHow much do you want to wager? "))
                if not 0 <= wager <= money:
                    wager = -1
                    print("Tried to fool me; bet again")
            except ValueError:
                print("Please enter a number.")
        # 获取玩家下注金额
        prompt = "\nIt's your move: "
        position = (1, 1, 1)
        while True:
            move = (-1, -1, -1)
            while move == (-1, -1, -1):
                try:
                    move = parse_move(input(prompt))
                except (ValueError, IndexError):
                    print("Please enter valid coordinates.")
            if (
                abs(move[0] - position[0])
                + abs(move[1] - position[1])
                + abs(move[2] - position[2])
            ) > 1:
                print("\nIllegal move. You lose")
                money = money - wager
                break
            elif (
                move[0] not in [1, 2, 3]
                or move[1] not in [1, 2, 3]
                or move[2] not in [1, 2, 3]
            ):
                print("\nIllegal move. You lose")
                money = money - wager
                break
            elif move == (3, 3, 3):
                print("\nCongratulations!")
                money = money + wager
                break
            elif move in mines:
                print("\n******BANG******")
                print("You lose!")
                money = money - wager
                break
            else:
                position = move
                prompt = "\nNext move: "
        # 进行游戏逻辑判断
        if money > 0:
            print("\nYou now have", money, "dollars.")
            if not input("Do you want to try again ").lower().startswith("y"):
                break
        else:
            print("\nYou bust.")
    print("\nTough luck")
    print("\nGoodbye.")
    # 打印游戏结果

def print_instructions() -> None:
    # 打印游戏说明
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
    # 主函数
    print(" " * 34 + "CUBE")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    if input("Do you want to see the instructions ").lower().startswith("y"):
        print_instructions()

    keep_playing = True
    while keep_playing:
        play_game()
        keep_playing = input("\nPlay again? (yes or no) ").lower().startswith("y")

if __name__ == "__main__":
    main()
# 如果作为脚本运行，则执行主函数

```