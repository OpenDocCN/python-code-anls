# `basic-computer-games\91_Train\python\train.py`

```
#!/usr/bin/env python3
# 设置脚本的解释器为 Python3

# TRAIN
#
# Converted from BASIC to Python by Trevor Hobson
# 程序的标题和作者信息

import random
# 导入 random 模块

def play_game() -> None:
    """Play one round of the game"""
    # 定义一个函数，用于进行一轮游戏
    car_speed = random.randint(40, 65)
    # 生成一个40到65之间的随机整数，表示汽车的速度
    time_difference = random.randint(5, 20)
    # 生成一个5到20之间的随机整数，表示时间差
    train_speed = random.randint(20, 39)
    # 生成一个20到39之间的随机整数，表示火车的速度
    print("\nA car travelling", car_speed, "MPH can make a certain trip in")
    print(time_difference, "hours less than a train travelling at", train_speed, "MPH")
    # 打印汽车和火车的速度信息
    time_answer: float = 0
    # 初始化时间答案为0
    while time_answer == 0:
        try:
            time_answer = float(input("How long does the trip take by car "))
            # 获取用户输入的汽车行程时间
        except ValueError:
            print("Please enter a number.")
            # 如果用户输入不是数字，则提示用户重新输入
    car_time = time_difference * train_speed / (car_speed - train_speed)
    # 计算汽车行程时间
    error_percent = int(abs((car_time - time_answer) * 100 / time_answer) + 0.5)
    # 计算误差百分比
    if error_percent > 5:
        print("Sorry. You were off by", error_percent, "percent.")
        print("Correct answer is", round(car_time, 6), "hours")
        # 如果误差大于5%，则提示用户答错，并给出正确答案
    else:
        print("Good! Answer within", error_percent, "percent.")
        # 如果误差小于等于5%，则提示用户答对

def main() -> None:
    # 定义主函数
    print(" " * 33 + "TRAIN")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    print("Time - speed distance exercise")
    # 打印游戏标题和信息

    keep_playing = True
    # 初始化继续游戏标志为 True
    while keep_playing:
        play_game()
        # 调用 play_game 函数进行游戏
        keep_playing = input("\nAnother problem (yes or no) ").lower().startswith("y")
        # 获取用户输入，判断是否继续游戏

if __name__ == "__main__":
    main()
    # 如果作为脚本直接执行，则调用主函数
```