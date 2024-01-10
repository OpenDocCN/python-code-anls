# `basic-computer-games\13_Bounce\python\bounce.py`

```
"""
BOUNCE

A physics simulation

Ported by Dave LeCompte
"""

from typing import Tuple, List

PAGE_WIDTH = 64


def print_centered(msg: str) -> None:
    # 计算需要添加的空格数，使得消息居中显示
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


def print_header(title: str) -> None:
    # 打印居中标题
    print_centered(title)
    # 打印居中副标题
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print()
    print()
    print()


def print_instructions() -> None:
    # 打印模拟说明
    print("THIS SIMULATION LETS YOU SPECIFY THE INITIAL VELOCITY")
    print("OF A BALL THROWN STRAIGHT UP, AND THE COEFFICIENT OF")
    print("ELASTICITY OF THE BALL.  PLEASE USE A DECIMAL FRACTION")
    print("COEFFICIENCY (LESS THAN 1).")
    print()
    print("YOU ALSO SPECIFY THE TIME INCREMENT TO BE USED IN")
    print("'STROBING' THE BALL'S FLIGHT (TRY .1 INITIALLY).")
    print()


def get_initial_conditions() -> Tuple[float, float, float]:
    # 获取模拟的初始条件
    delta_t = float(input("TIME INCREMENT (SEC)? "))
    print()
    v0 = float(input("VELOCITY (FPS)? "))
    print()
    coeff_rest = float(input("COEFFICIENT? "))
    print()

    return delta_t, v0, coeff_rest


def print_at_tab(line: str, tab: int, s: str) -> str:
    # 在指定位置添加字符串
    line += (" " * (tab - len(line))) + s
    return line


def run_simulation(delta_t: float, v0: float, coeff_rest: float) -> None:
    bounce_time: List[float] = [0] * 20  # time of each bounce

    print("FEET")
    print()

    sim_dur = int(70 / (v0 / (16 * delta_t)))
    for i in range(1, sim_dur + 1):
        bounce_time[i] = v0 * coeff_rest ** (i - 1) / 16

    # Draw the trajectory of the bouncing ball, one slice of height at a time
    h: float = int(-16 * (v0 / 32) ** 2 + v0**2 / 32 + 0.5)
    # 当高度大于等于0时，执行循环
    while h >= 0:
        # 初始化空字符串
        line = ""
        # 如果高度是整数，则将其转换为字符串并添加到line中
        if int(h) == h:
            line += str(int(h))
        # 初始化总时间为0
        total_time: float = 0
        # 遍历模拟持续时间范围
        for i in range(1, sim_dur + 1):
            # 初始化时间为0
            tm: float = 0
            # 当时间小于等于弹跳时间时执行循环
            while tm <= bounce_time[i]:
                # 总时间增加delta_t
                total_time += delta_t
                # 如果当前高度与计算出的高度差小于等于0.25，则在line中打印对应的时间和"0"
                if (
                    abs(h - (0.5 * (-32) * tm**2 + v0 * coeff_rest ** (i - 1) * tm))
                    <= 0.25
                ):
                    line = print_at_tab(line, int(total_time / delta_t), "0")
                # 时间增加delta_t
                tm += delta_t
            # 重置时间为下一个弹跳时间的一半
            tm = bounce_time[i + 1] / 2
            # 如果计算出的高度小于当前高度，则跳出循环
            if -16 * tm**2 + v0 * coeff_rest ** (i - 1) * tm < h:
                break
        # 打印line
        print(line)
        # 高度减去0.5
        h = h - 0.5

    # 打印"."，次数为总时间加1再除以delta_t再加1的整数部分
    print("." * (int((total_time + 1) / delta_t) + 1))
    # 打印空行
    print
    # 初始化line为" 0"
    line = " 0"
    # 遍历总时间加0.9995后的整数部分范围
    for i in range(1, int(total_time + 0.9995) + 1):
        # 在line中打印对应的时间和字符串形式的i
        line = print_at_tab(line, int(i / delta_t), str(i))
    # 打印line
    print(line)
    # 打印空行
    print()
    # 打印在空行中打印总时间加1再除以2倍的delta_t再减2的整数部分和"SECONDS"
    print(print_at_tab("", int((total_time + 1) / (2 * delta_t) - 2), "SECONDS"))
    # 打印空行
    print()
# 定义主函数，没有返回值
def main() -> None:
    # 打印标题
    print_header("BOUNCE")
    # 打印游戏说明
    print_instructions()

    # 无限循环，直到条件满足退出循环
    while True:
        # 获取初始条件：时间间隔、初始速度、恢复系数
        delta_t, v0, coeff_rest = get_initial_conditions()

        # 运行模拟，传入时间间隔、初始速度、恢复系数
        run_simulation(delta_t, v0, coeff_rest)
        # 退出循环
        break

# 如果当前脚本作为主程序运行，则执行主函数
if __name__ == "__main__":
    main()
```