# `30_Cube\python\cube.py`

```
#!/usr/bin/env python3  # 指定使用 Python3 解释器来执行脚本

"""
CUBE

Converted from BASIC to Python by Trevor Hobson
"""

import random  # 导入 random 模块，用于生成随机数
from typing import Tuple  # 从 typing 模块中导入 Tuple 类型，用于指定函数返回值的类型为元组


def mine_position() -> Tuple[int, int, int]:  # 定义函数 mine_position，返回类型为元组
    return (random.randint(1, 3), random.randint(1, 3), random.randint(1, 3))  # 返回一个包含三个随机整数的元组


def parse_move(move: str) -> Tuple[int, int, int]:  # 定义函数 parse_move，参数类型为字符串，返回类型为元组
    coordinates = [int(item) for item in move.split(",")]  # 将 move 字符串按逗号分割成列表，并将列表中的每个元素转换为整数
    if len(coordinates) == 3:  # 如果列表长度为3
        return tuple(coordinates)  # 返回包含列表元素的元组  # type: ignore
    raise ValueError  # 抛出值错误异常

def play_game() -> None:
    """Play one round of the game"""
    
    money = 500  # 初始化玩家的初始资金为500
    print("\nYou have", money, "dollars.")  # 打印玩家当前的资金
    while True:  # 进入游戏循环
        mines = []  # 初始化地雷列表
        for _ in range(5):  # 循环5次，生成5个地雷的位置
            while True:  # 进入循环，直到生成不重复的地雷位置
                mine = mine_position()  # 生成地雷的位置
                if not (mine in mines or mine == (1, 1, 1) or mine == (3, 3, 3)):  # 判断地雷位置是否重复或者与特定位置重复
                    break  # 如果位置不重复，则跳出循环
            mines.append(mine)  # 将地雷位置添加到地雷列表中
        wager = -1  # 初始化玩家的赌注为-1
        while wager == -1:  # 进入循环，直到玩家输入有效的赌注
            try:
                wager = int(input("\nHow much do you want to wager? "))  # 获取玩家输入的赌注金额
                if not 0 <= wager <= money:  # 如果下注金额不在合理范围内
                    wager = -1  # 将下注金额设为-1
                    print("Tried to fool me; bet again")  # 打印提示信息
            except ValueError:  # 捕获数值错误异常
                print("Please enter a number.")  # 打印提示信息
        prompt = "\nIt's your move: "  # 设置提示信息
        position = (1, 1, 1)  # 初始化位置坐标
        while True:  # 进入循环
            move = (-1, -1, -1)  # 初始化移动坐标
            while move == (-1, -1, -1):  # 当移动坐标为(-1, -1, -1)时
                try:  # 尝试执行以下代码
                    move = parse_move(input(prompt))  # 获取用户输入的移动坐标
                except (ValueError, IndexError):  # 捕获数值错误或索引错误异常
                    print("Please enter valid coordinates.")  # 打印提示信息
            if (
                abs(move[0] - position[0])  # 计算移动距离
                + abs(move[1] - position[1])
                + abs(move[2] - position[2])
            ) > 1:  # 如果移动距离大于1
                print("\nIllegal move. You lose")  # 打印提示信息
                money = money - wager  # 从玩家的资金中扣除赌注
                break  # 结束当前游戏循环
            elif (
                move[0] not in [1, 2, 3]  # 如果移动的第一个位置不在1、2、3中
                or move[1] not in [1, 2, 3]  # 或者移动的第二个位置不在1、2、3中
                or move[2] not in [1, 2, 3]  # 或者移动的第三个位置不在1、2、3中
            ):
                print("\nIllegal move. You lose")  # 打印出非法移动的消息，玩家输掉赌注
                money = money - wager  # 从玩家的资金中扣除赌注
                break  # 结束当前游戏循环
            elif move == (3, 3, 3):  # 如果移动的位置是(3, 3, 3)
                print("\nCongratulations!")  # 打印出祝贺消息
                money = money + wager  # 玩家赢得赌注
                break  # 结束当前游戏循环
            elif move in mines:  # 如果移动的位置在地雷的位置列表中
                print("\n******BANG******")  # 打印出爆炸的消息
                print("You lose!")  # 打印出玩家输掉赌注的消息
                money = money - wager  # 从玩家的资金中扣除赌注
                break  # 结束当前游戏循环
            else:  # 如果以上条件都不满足
                position = move  # 将变量 move 的值赋给变量 position
                prompt = "\nNext move: "  # 设置变量 prompt 的值为 "\nNext move: "
        if money > 0:  # 如果变量 money 的值大于 0
            print("\nYou now have", money, "dollars.")  # 打印当前拥有的金额
            if not input("Do you want to try again ").lower().startswith("y"):  # 如果用户输入的不是以 "y" 开头的字符串
                break  # 跳出循环
        else:  # 如果变量 money 的值不大于 0
            print("\nYou bust.")  # 打印信息
    print("\nTough luck")  # 打印信息
    print("\nGoodbye.")  # 打印信息


def print_instructions() -> None:  # 定义一个返回类型为 None 的函数 print_instructions
    print("\nThis is a game in which you will be playing against the")  # 打印游戏说明
    print("random decisions of the computer. The field of play is a")  # 打印游戏说明
    print("cube of side 3. Any of the 27 locations can be designated")  # 打印游戏说明
    print("by inputing three numbers such as 2,3,1. At the start,")  # 打印游戏说明
    print("you are automatically at location 1,1,1. The object of")  # 打印游戏说明
    print("the game is to get to location 3,3,3. One minor detail:")  # 打印游戏说明
    print("the computer will pick, at random, 5 locations at which")  # 打印游戏说明
    print("it will plant land mines. If you hit one of these locations")
    # 打印提示信息，说明游戏规则
    print("you lose. One other detail: You may move only one space")
    # 打印提示信息，说明游戏规则
    print("in one direction each move. For example: From 1,1,2 you")
    # 打印提示信息，说明游戏规则
    print("may move to 2,1,2 or 1,1,3. You may not change")
    # 打印提示信息，说明游戏规则
    print("two of the numbers on the same move. If you make an illegal")
    # 打印提示信息，说明游戏规则
    print("move, you lose and the computer takes the money you may")
    # 打印提示信息，说明游戏规则
    print("have bet on that round.\n")
    # 打印提示信息，说明游戏规则

    print("When stating the amount of a wager, print only the number")
    # 打印提示信息，说明下注金额的输入格式
    print("of dollars (example: 250) you are automatically started with")
    # 打印提示信息，说明初始金额
    print("500 dollars in your account.\n")
    # 打印提示信息，说明初始金额

    print("Good luck!")
    # 打印祝福信息


def main() -> None:
    print(" " * 34 + "CUBE")
    # 打印游戏标题
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    # 打印游戏信息

    if input("Do you want to see the instructions ").lower().startswith("y"):
        # 如果用户输入以"y"开头，则打印游戏说明
        print_instructions()

    keep_playing = True
    # 设置变量keep_playing为True，表示继续游戏
    while keep_playing:  # 当 keep_playing 为 True 时循环执行下面的代码
        play_game()  # 调用 play_game() 函数来进行游戏
        keep_playing = input("\nPlay again? (yes or no) ").lower().startswith("y")  # 获取用户输入，如果以 "y" 开头则将 keep_playing 设置为 True，否则设置为 False


if __name__ == "__main__":  # 如果当前脚本被直接执行，则执行下面的代码
    main()  # 调用 main() 函数来执行程序的主要逻辑
```