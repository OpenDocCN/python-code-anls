# `basic-computer-games\91_Train\python\train.py`

```

#!/usr/bin/env python3
# 指定解释器为 Python3

# TRAIN
#
# Converted from BASIC to Python by Trevor Hobson
# 基于 BASIC 语言转换为 Python，作者为 Trevor Hobson

import random
# 导入 random 模块

def play_game() -> None:
    """Play one round of the game"""
    # 定义一个函数，用于进行一轮游戏
    car_speed = random.randint(40, 65)
    # 生成一个40到65之间的随机整数，表示汽车速度
    time_difference = random.randint(5, 20)
    # 生成一个5到20之间的随机整数，表示时间差
    train_speed = random.randint(20, 39)
    # 生成一个20到39之间的随机整数，表示火车速度
    print("\nA car travelling", car_speed, "MPH can make a certain trip in")
    print(time_difference, "hours less than a train travelling at", train_speed, "MPH")
    # 打印汽车和火车的速度和时间差
    time_answer: float = 0
    # 初始化时间答案为0
    while time_answer == 0:
        try:
            time_answer = float(input("How long does the trip take by car "))
        except ValueError:
            print("Please enter a number.")
    # 循环直到输入一个数字为止
    car_time = time_difference * train_speed / (car_speed - train_speed)
    # 计算汽车行程时间
    error_percent = int(abs((car_time - time_answer) * 100 / time_answer) + 0.5)
    # 计算误差百分比
    if error_percent > 5:
        print("Sorry. You were off by", error_percent, "percent.")
        print("Correct answer is", round(car_time, 6), "hours")
    else:
        print("Good! Answer within", error_percent, "percent.")

def main() -> None:
    print(" " * 33 + "TRAIN")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    print("Time - speed distance exercise")
    # 打印游戏标题和介绍

    keep_playing = True
    while keep_playing:
        play_game()
        keep_playing = input("\nAnother problem (yes or no) ").lower().startswith("y")
    # 循环进行游戏，直到用户选择退出

if __name__ == "__main__":
    main()
    # 调用主函数

```