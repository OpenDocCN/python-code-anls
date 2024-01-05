# `13_Bounce\python\bounce.py`

```
"""
BOUNCE

A physics simulation

Ported by Dave LeCompte
"""

from typing import Tuple, List  # 导入类型提示模块，用于声明函数参数和返回值的类型

PAGE_WIDTH = 64  # 设置页面宽度为 64

# 定义一个函数，用于打印居中的消息
def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)  # 计算需要添加的空格数，使得消息居中显示
    print(spaces + msg)  # 打印居中的消息

# 定义一个函数，用于打印标题
def print_header(title: str) -> None:
    print_centered(title)  # 调用 print_centered 函数打印居中的标题
    # 打印居中的标题
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    # 打印空行
    print()
    print()
    print()


def print_instructions() -> None:
    # 打印模拟的说明
    print("THIS SIMULATION LETS YOU SPECIFY THE INITIAL VELOCITY")
    print("OF A BALL THROWN STRAIGHT UP, AND THE COEFFICIENT OF")
    print("ELASTICITY OF THE BALL.  PLEASE USE A DECIMAL FRACTION")
    print("COEFFICIENCY (LESS THAN 1).")
    print()
    print("YOU ALSO SPECIFY THE TIME INCREMENT TO BE USED IN")
    print("'STROBING' THE BALL'S FLIGHT (TRY .1 INITIALLY).")
    print()


def get_initial_conditions() -> Tuple[float, float, float]:
    # 获取用户输入的时间增量
    delta_t = float(input("TIME INCREMENT (SEC)? "))
    # 打印空行
    print()
    v0 = float(input("VELOCITY (FPS)? "))  # 从用户输入中获取速度值，并转换为浮点数
    print()  # 打印空行
    coeff_rest = float(input("COEFFICIENT? "))  # 从用户输入中获取系数值，并转换为浮点数
    print()  # 打印空行

    return delta_t, v0, coeff_rest  # 返回 delta_t, v0, coeff_rest


def print_at_tab(line: str, tab: int, s: str) -> str:
    line += (" " * (tab - len(line))) + s  # 在给定的行上添加空格，然后添加字符串 s
    return line  # 返回修改后的行


def run_simulation(delta_t: float, v0: float, coeff_rest: float) -> None:
    bounce_time: List[float] = [0] * 20  # 创建一个长度为 20 的浮点数列表，用于存储每次弹跳的时间

    print("FEET")  # 打印 "FEET"
    print()  # 打印空行

    sim_dur = int(70 / (v0 / (16 * delta_t)))  # 计算模拟持续时间并转换为整数
    for i in range(1, sim_dur + 1):
        # 计算每次弹跳的时间，并将结果存入字典中
        bounce_time[i] = v0 * coeff_rest ** (i - 1) / 16

    # 绘制弹球的轨迹，逐个高度切片绘制
    h: float = int(-16 * (v0 / 32) ** 2 + v0**2 / 32 + 0.5)
    while h >= 0:
        line = ""
        if int(h) == h:
            line += str(int(h))
        total_time: float = 0
        for i in range(1, sim_dur + 1):
            tm: float = 0
            while tm <= bounce_time[i]:
                total_time += delta_t
                # 如果当前高度与计算出的高度在一定范围内，则在该时间点绘制弹球位置
                if (
                    abs(h - (0.5 * (-32) * tm**2 + v0 * coeff_rest ** (i - 1) * tm))
                    <= 0.25
                ):
                    line = print_at_tab(line, int(total_time / delta_t), "0")
                tm += delta_t
            tm = bounce_time[i + 1] / 2  # 计算下一个弹跳时间的一半

            if -16 * tm**2 + v0 * coeff_rest ** (i - 1) * tm < h:  # 检查是否达到最大高度
                break  # 如果达到最大高度，跳出循环
        print(line)  # 打印当前行
        h = h - 0.5  # 更新高度

    print("." * (int((total_time + 1) / delta_t) + 1))  # 打印一行点，表示弹跳过程
    print  # 打印空行
    line = " 0"  # 初始化行
    for i in range(1, int(total_time + 0.9995) + 1):  # 遍历时间范围
        line = print_at_tab(line, int(i / delta_t), str(i))  # 在指定位置打印字符串
    print(line)  # 打印当前行
    print()  # 打印空行
    print(print_at_tab("", int((total_time + 1) / (2 * delta_t) - 2), "SECONDS"))  # 打印时间信息
    print()  # 打印空行


def main() -> None:
    print_header("BOUNCE")  # 打印标题
    print_instructions()  # 调用打印说明的函数，显示程序的说明或指导信息

    while True:  # 进入一个无限循环
        delta_t, v0, coeff_rest = get_initial_conditions()  # 调用获取初始条件的函数，获取时间间隔、初始速度和恢复系数

        run_simulation(delta_t, v0, coeff_rest)  # 调用运行模拟的函数，使用获取的初始条件运行模拟
        break  # 跳出循环

if __name__ == "__main__":  # 如果当前脚本被直接执行
    main()  # 调用主函数
```