# `86_Target\python\target.py`

```
"""
TARGET

Weapon targeting simulation / 3d trigonometry practice

Ported by Dave LeCompte
"""

import math  # 导入数学库
import random  # 导入随机数库
from typing import List  # 从 typing 模块中导入 List 类型

PAGE_WIDTH = 64  # 设置页面宽度为 64

# 定义一个函数，用于打印居中的消息
def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)  # 计算需要添加的空格数，使得消息居中显示
    print(spaces + msg)  # 打印居中的消息
# 打印标题和作者信息
def print_header(title: str) -> None:
    print_centered(title)  # 调用打印居中函数，打印标题
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 调用打印居中函数，打印作者信息
    print()  # 打印空行
    print()  # 打印空行
    print()  # 打印空行


# 打印游戏说明
def print_instructions() -> None:
    print("YOU ARE THE WEAPONS OFFICER ON THE STARSHIP ENTERPRISE")  # 打印游戏角色信息
    print("AND THIS IS A TEST TO SEE HOW ACCURATE A SHOT YOU")  # 打印游戏目的
    print("ARE IN A THREE-DIMENSIONAL RANGE.  YOU WILL BE TOLD")  # 打印游戏说明
    print("THE RADIAN OFFSET FOR THE X AND Z AXES, THE LOCATION")  # 打印游戏说明
    print("OF THE TARGET IN THREE DIMENSIONAL RECTANGULAR COORDINATES,")  # 打印游戏说明
    print("THE APPROXIMATE NUMBER OF DEGREES FROM THE X AND Z")  # 打印游戏说明
    print("AXES, AND THE APPROXIMATE DISTANCE TO THE TARGET.")  # 打印游戏说明
    print("YOU WILL THEN PROCEEED TO SHOOT AT THE TARGET UNTIL IT IS")  # 打印游戏说明
    print("DESTROYED!")  # 打印游戏说明
    print()  # 打印空行
    print("GOOD LUCK!!")  # 打印祝福信息
    print()  # 打印空行
    print()  # 打印空行


def prompt() -> List[float]:
    while True:
        response = input("INPUT ANGLE DEVIATION FROM X, DEVIATION FROM Z, DISTANCE? ")  # 提示用户输入角度偏差、Z轴偏差和距离
        if "," not in response:  # 如果输入中不包含逗号，则继续循环
            continue

        terms = response.split(",")  # 将输入按逗号分割成列表
        if len(terms) != 3:  # 如果分割后的列表长度不为3，则继续循环
            continue

        return [float(t) for t in terms]  # 将分割后的列表中的每个元素转换为浮点数并返回


def next_target() -> None:
    for _ in range(5):  # 循环5次
        print()  # 打印空行
    print(f"SHOT BELOW TARGET {-z2:.2f} KILOMETERS.")
    else:
        print(f"SHOT ABOVE TARGET {z2:.2f} KILOMETERS.")
        print(f"SHOT BELOW TARGET {-z2:.2f} KILOMETERS.")  # 打印距离目标下方的距离
    else:
        print(f"SHOT ABOVE TARGET {z2:.2f} KILOMETERS.")  # 打印距离目标上方的距离

    print(f"APPROX POSITION OF EXPLOSION:  X={x1:.4f}   Y={y1:.4f}   Z={z1:.4f}")  # 打印爆炸的大致位置坐标
    print(f"     DISTANCE FROM TARGET = {d:.2f}")  # 打印距离目标的距离
    print()  # 打印空行
    print()  # 打印空行
    print()  # 打印空行


def do_shot_loop(p1, x, y, z) -> None:
    shot_count = 0  # 初始化射击次数为0
    while True:  # 进入无限循环
        shot_count += 1  # 射击次数加1
        if shot_count == 1:  # 如果射击次数为1
            p3 = int(p1 * 0.05) * 20  # 计算p3的值
        elif shot_count == 2:  # 如果射击次数为2
            p3 = int(p1 * 0.1) * 10  # 计算p3的值
        elif shot_count == 3:  # 如果射击次数为3
        p3 = int(p1 * 0.5) * 2  # 将p1乘以0.5，然后取整数部分，再乘以2，赋值给p3
        elif shot_count == 4:  # 如果shot_count等于4
            p3 = int(p1)  # 将p1取整数部分，赋值给p3
        else:  # 其他情况
            p3 = p1  # 将p1赋值给p3

        if p3 == int(p3):  # 如果p3等于其整数部分
            print(f"     ESTIMATED DISTANCE: {p3}")  # 打印估计距离的整数部分
        else:  # 否则
            print(f"     ESTIMATED DISTANCE: {p3:.2f}")  # 打印估计距离的小数部分，保留两位小数
        print()  # 打印空行
        a1, b1, p2 = prompt()  # 从用户输入中获取a1、b1、p2的值

        if p2 < 20:  # 如果p2小于20
            print("YOU BLEW YOURSELF UP!!")  # 打印提示信息
            return  # 结束函数

        a1 = math.radians(a1)  # 将a1转换为弧度
        b1 = math.radians(b1)  # 将b1转换为弧度
        show_radians(a1, b1)  # 调用show_radians函数，传入a1和b1作为参数
        x1 = p2 * math.sin(b1) * math.cos(a1)  # 计算目标点的 x 坐标
        y1 = p2 * math.sin(b1) * math.sin(a1)  # 计算目标点的 y 坐标
        z1 = p2 * math.cos(b1)  # 计算目标点的 z 坐标

        distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2 + (z1 - z) ** 2)  # 计算炮弹与目标点的距离

        if distance <= 20:  # 如果距离小于等于20
            print()  # 打印空行
            print(" * * * HIT * * *   TARGET IS NON FUNCTIONAL")  # 打印命中目标的提示
            print()  # 打印空行
            print(f"DISTANCE OF EXPLOSION FROM TARGET WAS {distance:.4f} KILOMETERS")  # 打印爆炸距离目标的距离
            print()  # 打印空行
            print(f"MISSION ACCOMPLISHED IN {shot_count} SHOTS.")  # 打印完成任务所用的炮弹数量
            return  # 返回

        else:  # 如果距离大于20
            describe_miss(x, y, z, x1, y1, z1, distance)  # 调用函数描述未命中情况
def show_radians(a, b) -> None:
    # 打印从 X 轴和 Z 轴的弧度值
    print(f"RADIANS FROM X AXIS = {a:.4f}   FROM Z AXIS = {b:.4f}")


def play_game() -> None:
    while True:
        a = random.uniform(0, 2 * math.pi)  # 随机角度
        b = random.uniform(0, 2 * math.pi)  # 随机角度

        show_radians(a, b)  # 调用函数显示弧度值

        p1 = random.uniform(0, 100000) + random.uniform(0, 1)  # 随机生成一个数值
        x = math.sin(b) * math.cos(a) * p1  # 根据公式计算 x 坐标
        y = math.sin(b) * math.sin(a) * p1  # 根据公式计算 y 坐标
        z = math.cos(b) * p1  # 根据公式计算 z 坐标
        print(
            f"TARGET SIGHTED: APPROXIMATE COORDINATES:  X={x:.1f}  Y={y:.1f}  Z={z:.1f}"
        )  # 打印目标坐标

        do_shot_loop(p1, x, y, z)  # 调用射击循环函数
        next_target()  # 调用函数 next_target()，用于获取下一个目标

def main() -> None:
    print_header("TARGET")  # 调用函数 print_header()，打印标题 "TARGET"
    print_instructions()  # 调用函数 print_instructions()，打印游戏说明

    play_game()  # 调用函数 play_game()，开始游戏

if __name__ == "__main__":
    main()  # 如果当前脚本被直接执行，则调用函数 main()，开始执行游戏
```