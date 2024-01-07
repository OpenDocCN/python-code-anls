# `basic-computer-games\86_Target\python\target.py`

```

"""
TARGET

Weapon targeting simulation / 3d trigonometry practice

Ported by Dave LeCompte
"""

import math  # 导入数学库
import random  # 导入随机数库
from typing import List  # 导入类型提示库

PAGE_WIDTH = 64  # 页面宽度常量


def print_centered(msg: str) -> None:  # 定义打印居中函数，参数为字符串，返回空
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)  # 计算空格数
    print(spaces + msg)  # 打印居中的消息


def print_header(title: str) -> None:  # 定义打印标题函数，参数为字符串，返回空
    print_centered(title)  # 打印居中的标题
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 打印居中的副标题
    print()  # 打印空行
    print()  # 打印空行
    print()  # 打印空行


def print_instructions() -> None:  # 定义打印说明函数，返回空
    # 打印游戏说明
    print("YOU ARE THE WEAPONS OFFICER ON THE STARSHIP ENTERPRISE")
    print("AND THIS IS A TEST TO SEE HOW ACCURATE A SHOT YOU")
    print("ARE IN A THREE-DIMENSIONAL RANGE.  YOU WILL BE TOLD")
    print("THE RADIAN OFFSET FOR THE X AND Z AXES, THE LOCATION")
    print("OF THE TARGET IN THREE DIMENSIONAL RECTANGULAR COORDINATES,")
    print("THE APPROXIMATE NUMBER OF DEGREES FROM THE X AND Z")
    print("AXES, AND THE APPROXIMATE DISTANCE TO THE TARGET.")
    print("YOU WILL THEN PROCEEED TO SHOOT AT THE TARGET UNTIL IT IS")
    print("DESTROYED!")
    print()
    print("GOOD LUCK!!")
    print()
    print()


def prompt() -> List[float]:  # 定义提示函数，返回浮点数列表
    while True:  # 无限循环
        response = input("INPUT ANGLE DEVIATION FROM X, DEVIATION FROM Z, DISTANCE? ")  # 提示用户输入
        if "," not in response:  # 如果输入不包含逗号
            continue  # 继续循环

        terms = response.split(",")  # 以逗号分割输入
        if len(terms) != 3:  # 如果分割后的列表长度不为3
            continue  # 继续循环

        return [float(t) for t in terms]  # 返回转换为浮点数的列表


def next_target() -> None:  # 定义下一个目标函数，返回空
    for _ in range(5):  # 循环5次
        print()  # 打印空行
    print("NEXT TARGET...")  # 打印下一个目标提示
    print()  # 打印空行


def describe_miss(x, y, z, x1, y1, z1, d) -> None:  # 定义描述未命中函数，参数为坐标和距离，返回空
    x2 = x1 - x  # 计算x坐标偏差
    y2 = y1 - y  # 计算y坐标偏差
    z2 = z1 - z  # 计算z坐标偏差

    if x2 < 0:  # 如果x坐标偏差小于0
        print(f"SHOT BEHIND TARGET {-x2:.2f} KILOMETERS.")  # 打印在目标后方的提示
    else:  # 否则
        print(f"SHOT IN FRONT OF TARGET {x2:.2f} KILOMETERS.")  # 打印在目标前方的提示

    if y2 < 0:  # 如果y坐标偏差小于0
        print(f"SHOT TO RIGHT OF TARGET {-y2:.2f} KILOMETERS.")  # 打印在目标右侧的提示
    else:  # 否则
        print(f"SHOT TO LEFT OF TARGET {y2:.2f} KILOMETERS.")  # 打印在目标左侧的提示

    if z2 < 0:  # 如果z坐标偏差小于0
        print(f"SHOT BELOW TARGET {-z2:.2f} KILOMETERS.")  # 打印在目标下方的提示
    else:  # 否则
        print(f"SHOT ABOVE TARGET {z2:.2f} KILOMETERS.")  # 打印在目标上方的提示

    print(f"APPROX POSITION OF EXPLOSION:  X={x1:.4f}   Y={y1:.4f}   Z={z1:.4f}")  # 打印爆炸位置的坐标
    print(f"     DISTANCE FROM TARGET = {d:.2f}")  # 打印距离目标的距离
    print()  # 打印空行
    print()  # 打印空行
    print()  # 打印空行


def do_shot_loop(p1, x, y, z) -> None:  # 定义射击循环函数，参数为角度和坐标，返回空
    shot_count = 0  # 初始化射击次数为0
    while True:  # 无限循环
        shot_count += 1  # 射击次数加1
        if shot_count == 1:  # 如果射击次数为1
            p3 = int(p1 * 0.05) * 20  # 计算估计距离
        elif shot_count == 2:  # 如果射击次数为2
            p3 = int(p1 * 0.1) * 10  # 计算估计距离
        elif shot_count == 3:  # 如果射击次数为3
            p3 = int(p1 * 0.5) * 2  # 计算估计距离
        elif shot_count == 4:  # 如果射击次数为4
            p3 = int(p1)  # 计算估计距离
        else:  # 否则
            p3 = p1  # 估计距离为p1

        if p3 == int(p3):  # 如果估计距离为整数
            print(f"     ESTIMATED DISTANCE: {p3}")  # 打印估计距离
        else:  # 否则
            print(f"     ESTIMATED DISTANCE: {p3:.2f}")  # 打印估计距离
        print()  # 打印空行
        a1, b1, p2 = prompt()  # 获取用户输入的角度和距离

        if p2 < 20:  # 如果距离小于20
            print("YOU BLEW YOURSELF UP!!")  # 打印自爆提示
            return  # 返回

        a1 = math.radians(a1)  # 将角度转换为弧度
        b1 = math.radians(b1)  # 将角度转换为弧度
        show_radians(a1, b1)  # 显示弧度

        x1 = p2 * math.sin(b1) * math.cos(a1)  # 计算x坐标
        y1 = p2 * math.sin(b1) * math.sin(a1)  # 计算y坐标
        z1 = p2 * math.cos(b1)  # 计算z坐标

        distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2 + (z1 - z) ** 2)  # 计算距离

        if distance <= 20:  # 如果距离小于等于20
            print()  # 打印空行
            print(" * * * HIT * * *   TARGET IS NON FUNCTIONAL")  # 打印命中提示
            print()  # 打印空行
            print(f"DISTANCE OF EXPLOSION FROM TARGET WAS {distance:.4f} KILOMETERS")  # 打印爆炸距离
            print()  # 打印空行
            print(f"MISSION ACCOMPLISHED IN {shot_count} SHOTS.")  # 打印完成任务的射击次数

            return  # 返回
        else:  # 否则
            describe_miss(x, y, z, x1, y1, z1, distance)  # 描述未命中情况


def show_radians(a, b) -> None:  # 定义显示弧度函数，参数为a和b，返回空
    print(f"RADIANS FROM X AXIS = {a:.4f}   FROM Z AXIS = {b:.4f}")  # 打印从X轴和Z轴的弧度


def play_game() -> None:  # 定义游戏函数，返回空
    while True:  # 无限循环
        a = random.uniform(0, 2 * math.pi)  # 随机生成角度
        b = random.uniform(0, 2 * math.pi)  # 随机生成角度

        show_radians(a, b)  # 显示弧度

        p1 = random.uniform(0, 100000) + random.uniform(0, 1)  # 随机生成距离
        x = math.sin(b) * math.cos(a) * p1  # 计算x坐标
        y = math.sin(b) * math.sin(a) * p1  # 计算y坐标
        z = math.cos(b) * p1  # 计算z坐标
        print(
            f"TARGET SIGHTED: APPROXIMATE COORDINATES:  X={x:.1f}  Y={y:.1f}  Z={z:.1f}"
        )  # 打印目标坐标

        do_shot_loop(p1, x, y, z)  # 进行射击循环
        next_target()  # 进入下一个目标


def main() -> None:  # 定义主函数，返回空
    print_header("TARGET")  # 打印标题
    print_instructions()  # 打印说明

    play_game()  # 进行游戏


if __name__ == "__main__":  # 如果是主程序入口
    main()  # 调用主函数

```