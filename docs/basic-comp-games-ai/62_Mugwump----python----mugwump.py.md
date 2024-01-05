# `62_Mugwump\python\mugwump.py`

```
from math import sqrt  # 导入 sqrt 函数，用于计算平方根
from random import randint  # 导入 randint 函数，用于生成随机整数
from typing import List, Tuple  # 导入 List 和 Tuple 类型提示，用于函数参数和返回值的类型注解


def introduction() -> None:
    print(
        """The object of this game is to find 4 mugwumps
hidden on a 10*10 grid.  Homebase is position 0,0.
Any guess you make must be two numbers with each
number between 0 and 9 inclusive.  First number
is distance to right of homebase, and second number
is the distance above homebase."""
    )  # 打印游戏介绍信息

    print(
        """You get 10 tries.  After each try, I will tell
you how far you are from each mugwump."""
    )  # 打印游戏规则信息
# 生成指定数量的Mugwumps的坐标
def generate_mugwumps(n: int = 4) -> List[List[int]]:
    mugwumps = []
    for _ in range(n):
        current = [randint(0, 9), randint(0, 9)]  # 生成随机的Mugwump坐标
        mugwumps.append(current)  # 将生成的坐标添加到mugwumps列表中
    return mugwumps  # 返回生成的Mugwumps坐标列表


# 显示Mugwumps的位置
def reveal_mugwumps(mugwumps: List[List[int]]) -> None:
    print("Sorry, that's 10 tries.  Here's where they're hiding.")  # 打印提示信息
    for idx, mugwump in enumerate(mugwumps, 1):  # 遍历Mugwumps列表
        if mugwump[0] != -1:  # 如果Mugwump的横坐标不为-1
            print(f"Mugwump {idx} is at {mugwump[0]},{mugwump[1]}")  # 打印Mugwump的位置信息


# 计算猜测点与Mugwump之间的距离
def calculate_distance(guess: Tuple[int, int], mugwump: List[int]) -> float:
    d = sqrt(((mugwump[0] - guess[0]) ** 2) + ((mugwump[1] - guess[1]) ** 2))  # 计算欧几里得距离
    return d  # 返回计算得到的距离
def play_again() -> None:
    # 打印提示信息
    print("THAT WAS FUN! LET'S PLAY AGAIN.......")
    # 获取用户输入
    choice = input("Press Enter to play again, any other key then Enter to quit.")
    # 判断用户输入，如果是空字符串则继续游戏，否则退出游戏
    if choice == "":
        print("Four more mugwumps are now in hiding.")
    else:
        exit()


def play_round() -> None:
    # 生成 mugwumps
    mugwumps = generate_mugwumps()
    # 初始化轮数和得分
    turns = 1
    score = 0
    # 循环进行游戏，最多进行10轮或者猜对4个 mugwumps 就结束游戏
    while turns <= 10 and score != 4:
        # 初始化 m
        m = -1
        # 循环直到 m 不等于 -1
        while m == -1:
            try:
                # 获取用户输入并转换为整数
                m, n = map(int, input(f"Turn {turns} - what is your guess? ").split())
            except ValueError:
                m = -1  # 初始化变量 m 为 -1
        for idx, mugwump in enumerate(mugwumps):  # 遍历 mugwumps 列表，同时获取索引和值
            if m == mugwump[0] and n == mugwump[1]:  # 如果 m 和 n 分别等于 mugwump 列表中的第一个和第二个元素
                print(f"You found mugwump {idx + 1}")  # 打印找到的 mugwump 的索引
                mugwumps[idx][0] = -1  # 将找到的 mugwump 的第一个元素设为 -1
                score += 1  # 分数加一
            if mugwump[0] == -1:  # 如果 mugwump 的第一个元素为 -1
                continue  # 继续下一次循环
            print(
                f"You are {calculate_distance((m, n), mugwump):.1f} units from mugwump {idx + 1}"
            )  # 打印距离信息
        turns += 1  # 回合数加一
    if score == 4:  # 如果分数等于 4
        print(f"Well done! You got them all in {turns} turns.")  # 打印恭喜信息和回合数
    else:
        reveal_mugwumps(mugwumps)  # 调用 reveal_mugwumps 函数，传入 mugwumps 列表


if __name__ == "__main__":
    introduction()  # 调用 introduction 函数
    while True:  # 创建一个无限循环
        play_round()  # 调用 play_round() 函数，执行游戏的一个回合
        play_again()  # 调用 play_again() 函数，询问玩家是否要再玩一次
```