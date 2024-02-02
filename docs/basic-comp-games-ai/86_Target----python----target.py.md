# `basic-computer-games\86_Target\python\target.py`

```py
"""
TARGET

Weapon targeting simulation / 3d trigonometry practice

Ported by Dave LeCompte
"""

# 导入数学库
import math
# 导入随机数库
import random
# 导入类型提示库
from typing import List

# 页面宽度常量
PAGE_WIDTH = 64


# 打印居中文本
def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


# 打印标题
def print_header(title: str) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print()
    print()
    print()


# 打印游戏说明
def print_instructions() -> None:
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


# 提示用户输入角度偏差和距离
def prompt() -> List[float]:
    while True:
        response = input("INPUT ANGLE DEVIATION FROM X, DEVIATION FROM Z, DISTANCE? ")
        if "," not in response:
            continue

        terms = response.split(",")
        if len(terms) != 3:
            continue

        return [float(t) for t in terms]


# 下一个目标
def next_target() -> None:
    for _ in range(5):
        print()
    print("NEXT TARGET...")
    print()


# 描述未命中情况
def describe_miss(x, y, z, x1, y1, z1, d) -> None:
    x2 = x1 - x
    y2 = y1 - y
    z2 = z1 - z

    if x2 < 0:
        print(f"SHOT BEHIND TARGET {-x2:.2f} KILOMETERS.")
    else:
        print(f"SHOT IN FRONT OF TARGET {x2:.2f} KILOMETERS.")

    if y2 < 0:
        print(f"SHOT TO RIGHT OF TARGET {-y2:.2f} KILOMETERS.")
    else:
        print(f"SHOT TO LEFT OF TARGET {y2:.2f} KILOMETERS.")

    if z2 < 0:
        print(f"SHOT BELOW TARGET {-z2:.2f} KILOMETERS.")
    # 如果未命中目标，则打印出射击点高于目标的信息
    else:
        print(f"SHOT ABOVE TARGET {z2:.2f} KILOMETERS.")

    # 打印爆炸的大致位置的 X、Y、Z 坐标
    print(f"APPROX POSITION OF EXPLOSION:  X={x1:.4f}   Y={y1:.4f}   Z={z1:.4f}")
    # 打印距离目标的距离
    print(f"     DISTANCE FROM TARGET = {d:.2f}")
    # 打印空行
    print()
    # 打印空行
    print()
    # 打印空行
    print()
# 定义一个函数，用于进行射击循环
def do_shot_loop(p1, x, y, z) -> None:
    # 初始化射击次数
    shot_count = 0
    # 进入无限循环
    while True:
        # 射击次数加一
        shot_count += 1
        # 根据射击次数选择不同的计算方式得到 p3
        if shot_count == 1:
            p3 = int(p1 * 0.05) * 20
        elif shot_count == 2:
            p3 = int(p1 * 0.1) * 10
        elif shot_count == 3:
            p3 = int(p1 * 0.5) * 2
        elif shot_count == 4:
            p3 = int(p1)
        else:
            p3 = p1

        # 判断 p3 是否为整数，根据结果打印不同的信息
        if p3 == int(p3):
            print(f"     ESTIMATED DISTANCE: {p3}")
        else:
            print(f"     ESTIMATED DISTANCE: {p3:.2f}")
        print()
        # 获取用户输入的角度和距离
        a1, b1, p2 = prompt()

        # 如果距离小于 20，打印信息并结束函数
        if p2 < 20:
            print("YOU BLEW YOURSELF UP!!")
            return

        # 将角度转换为弧度
        a1 = math.radians(a1)
        b1 = math.radians(b1)
        # 打印转换后的弧度
        show_radians(a1, b1)

        # 根据用户输入的角度和距离计算新的坐标
        x1 = p2 * math.sin(b1) * math.cos(a1)
        y1 = p2 * math.sin(b1) * math.sin(a1)
        z1 = p2 * math.cos(b1)

        # 计算新坐标与目标坐标的距离
        distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2 + (z1 - z) ** 2)

        # 如果距离小于等于 20，打印信息并结束函数
        if distance <= 20:
            print()
            print(" * * * HIT * * *   TARGET IS NON FUNCTIONAL")
            print()
            print(f"DISTANCE OF EXPLOSION FROM TARGET WAS {distance:.4f} KILOMETERS")
            print()
            print(f"MISSION ACCOMPLISHED IN {shot_count} SHOTS.")
            return
        # 否则打印未命中的描述信息
        else:
            describe_miss(x, y, z, x1, y1, z1, distance)


# 定义一个函数，用于打印弧度信息
def show_radians(a, b) -> None:
    print(f"RADIANS FROM X AXIS = {a:.4f}   FROM Z AXIS = {b:.4f}")


# 定义一个函数，用于进行游戏
def play_game() -> None:
    # 进入无限循环
    while True:
        # 生成随机的角度
        a = random.uniform(0, 2 * math.pi)  # random angle
        b = random.uniform(0, 2 * math.pi)  # random angle

        # 打印随机生成的角度的弧度信息
        show_radians(a, b)

        # 生成随机的 p1 和对应的坐标
        p1 = random.uniform(0, 100000) + random.uniform(0, 1)
        x = math.sin(b) * math.cos(a) * p1
        y = math.sin(b) * math.sin(a) * p1
        z = math.cos(b) * p1
        print(
            f"TARGET SIGHTED: APPROXIMATE COORDINATES:  X={x:.1f}  Y={y:.1f}  Z={z:.1f}"
        )

        # 调用射击循环函数
        do_shot_loop(p1, x, y, z)
        # 进入下一个目标
        next_target()
# 定义主函数，没有返回值
def main() -> None:
    # 打印标题
    print_header("TARGET")
    # 打印游戏说明
    print_instructions()

    # 开始游戏
    play_game()

# 如果当前脚本作为主程序执行，则调用主函数
if __name__ == "__main__":
    main()
```