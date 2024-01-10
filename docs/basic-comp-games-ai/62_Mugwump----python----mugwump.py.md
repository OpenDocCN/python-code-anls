# `basic-computer-games\62_Mugwump\python\mugwump.py`

```
# 从 math 模块中导入 sqrt 函数
from math import sqrt
# 从 random 模块中导入 randint 函数
from random import randint
# 从 typing 模块中导入 List 和 Tuple 类型
from typing import List, Tuple

# 定义一个无返回值的函数，用于介绍游戏规则
def introduction() -> None:
    # 打印游戏规则介绍
    print(
        """The object of this game is to find 4 mugwumps
hidden on a 10*10 grid.  Homebase is position 0,0.
Any guess you make must be two numbers with each
number between 0 and 9 inclusive.  First number
is distance to right of homebase, and second number
is the distance above homebase."""
    )
    # 打印游戏提示信息
    print(
        """You get 10 tries.  After each try, I will tell
you how far you are from each mugwump."""
    )

# 定义一个生成 mugwumps 位置的函数，返回一个包含 mugwumps 位置的列表
def generate_mugwumps(n: int = 4) -> List[List[int]]:
    # 初始化一个空列表用于存放 mugwumps 位置
    mugwumps = []
    # 循环生成 n 个 mugwumps 的位置
    for _ in range(n):
        # 生成一个随机的 mugwump 位置
        current = [randint(0, 9), randint(0, 9)]
        # 将生成的位置添加到 mugwumps 列表中
        mugwumps.append(current)
    # 返回包含 mugwumps 位置的列表
    return mugwumps

# 定义一个无返回值的函数，用于展示 mugwumps 的位置
def reveal_mugwumps(mugwumps: List[List[int]]) -> None:
    # 打印提示信息
    print("Sorry, that's 10 tries.  Here's where they're hiding.")
    # 遍历 mugwumps 列表，打印每个 mugwump 的位置
    for idx, mugwump in enumerate(mugwumps, 1):
        # 如果 mugwump 的位置不是 [-1, -1]，则打印其位置
        if mugwump[0] != -1:
            print(f"Mugwump {idx} is at {mugwump[0]},{mugwump[1]}")

# 定义一个计算猜测位置与 mugwump 位置之间距离的函数，返回距离值
def calculate_distance(guess: Tuple[int, int], mugwump: List[int]) -> float:
    # 计算距离值
    d = sqrt(((mugwump[0] - guess[0]) ** 2) + ((mugwump[1] - guess[1]) ** 2))
    # 返回距离值
    return d

# 定义一个无返回值的函数，用于询问是否再玩一次游戏
def play_again() -> None:
    # 打印提示信息
    print("THAT WAS FUN! LET'S PLAY AGAIN.......")
    # 获取用户输入
    choice = input("Press Enter to play again, any other key then Enter to quit.")
    # 根据用户输入决定是否再玩一次游戏
    if choice == "":
        print("Four more mugwumps are now in hiding.")
    else:
        exit()

# 定义一个无返回值的函数，用于进行一轮游戏
def play_round() -> None:
    # 生成 mugwumps 的位置
    mugwumps = generate_mugwumps()
    # 初始化回合数
    turns = 1
    # 初始化得分
    score = 0
    # 当轮数小于等于10且得分不等于4时，执行循环
    while turns <= 10 and score != 4:
        # 初始化m为-1
        m = -1
        # 当m为-1时，循环执行以下代码，直到输入正确的值
        while m == -1:
            try:
                # 从用户输入中获取两个整数值
                m, n = map(int, input(f"Turn {turns} - what is your guess? ").split())
            except ValueError:
                # 如果输入不是整数，将m重新设置为-1
                m = -1
        # 遍历mugwumps列表
        for idx, mugwump in enumerate(mugwumps):
            # 如果用户猜测的坐标与mugwump的坐标匹配
            if m == mugwump[0] and n == mugwump[1]:
                # 打印找到mugwump的消息
                print(f"You found mugwump {idx + 1}")
                # 将找到的mugwump的坐标标记为-1
                mugwumps[idx][0] = -1
                # 增加得分
                score += 1
            # 如果mugwump的坐标已经被找到
            if mugwump[0] == -1:
                # 继续下一次循环
                continue
            # 打印用户距离mugwump的距离
            print(
                f"You are {calculate_distance((m, n), mugwump):.1f} units from mugwump {idx + 1}"
            )
        # 增加轮数
        turns += 1
    # 如果得分等于4
    if score == 4:
        # 打印猜测成功的消息
        print(f"Well done! You got them all in {turns} turns.")
    else:
        # 显示mugwumps的位置
        reveal_mugwumps(mugwumps)
# 如果当前模块被直接执行，则执行以下代码
if __name__ == "__main__":
    # 调用介绍函数，输出游戏介绍
    introduction()
    # 无限循环，直到游戏结束
    while True:
        # 执行游戏回合
        play_round()
        # 询问玩家是否再玩一次
        play_again()
```