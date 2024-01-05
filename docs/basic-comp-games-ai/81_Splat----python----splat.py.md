# `81_Splat\python\splat.py`

```
"""
SPLAT

Splat similates a parachute jump in which you try to open your parachute
at the last possible moment without going splat! You may select your own
terminal velocity or let the computer do it for you. You may also select
the acceleration due to gravity or, again, let the computer do it
in which case you might wind up on any one of the eight planets (out to
Neptune), the moon, or the sun.

The computer then tells you the height you're jumping from and asks for
the seconds of free fall. It then divides your free fall time into eight
intervals and gives you progress reports on the way down. The computer
also keeps track of all prior jumps and lets you know how you compared
with previous successful jumps. If you want to recall information from
previous runs, then you should store the array `successful_jumps` on
disk and read it before each run.

John Yegge created this program while at the Oak Ridge Associated
Universities.
"""
Ported in 2021 by Jonas Nockert / @lemonad
# 代码作者和修改信息的注释

"""
from math import sqrt
from random import choice, random, uniform
from typing import List, Tuple

PAGE_WIDTH = 72
# 导入所需的模块和类型提示

def numeric_input(question, default=0) -> float:
    """Ask user for a numeric value."""
    # 定义一个函数，用于询问用户输入一个数值
    while True:
        answer_str = input(f"{question} [{default}]: ").strip() or default
        # 询问用户输入一个数值，并将其转换为浮点数
        try:
            return float(answer_str)
        except ValueError:
            pass
        # 如果用户输入的不是数值，则继续询问
def yes_no_input(question: str, default="YES") -> bool:
    """Ask user a yes/no question and returns True if yes, otherwise False."""
    # 询问用户一个yes/no问题，并且如果是yes则返回True，否则返回False
    answer = input(f"{question} (YES OR NO) [{default}]: ").strip() or default
    while answer.lower() not in ["n", "no", "y", "yes"]:
        answer = input(f"YES OR NO [{default}]: ").strip() or default
    return answer.lower() in ["y", "yes"]


def get_terminal_velocity() -> float:
    """Terminal velocity by user or picked by computer."""
    # 用户选择自己的终端速度或者由计算机选择
    if yes_no_input("SELECT YOUR OWN TERMINAL VELOCITY", default="NO"):
        v1 = numeric_input("WHAT TERMINAL VELOCITY (MI/HR)", default=100)
    else:
        # 计算机选择0-1000的终端速度
        v1 = int(1000 * random())
        print(f"OK.  TERMINAL VELOCITY = {v1} MI/HR")

    # 将英里/小时转换为英尺/秒
    return v1 * (5280 / 3600)
def get_acceleration() -> float:
    """Acceleration due to gravity by user or picked by computer."""
    # 如果用户想要选择重力加速度
    if yes_no_input("WANT TO SELECT ACCELERATION DUE TO GRAVITY", default="NO"):
        # 用户输入重力加速度
        a2 = numeric_input("WHAT ACCELERATION (FT/SEC/SEC)", default=32.16)
    else:
        # 计算机随机选择天体并返回其重力加速度
        body, a2 = pick_random_celestial_body()
        print(f"FINE. YOU'RE ON {body}. ACCELERATION={a2} FT/SEC/SEC.")
    return a2


def get_freefall_time() -> float:
    """User-guessed freefall time.

    The idea of the game is to pick a freefall time, given initial
    altitude, terminal velocity and acceleration, so the parachute
    as close to the ground as possible without going splat.
    """
    # 初始化自由落体时间
    t_freefall: float = 0
    # A zero or negative freefall time is not handled by the motion
    # equations during the jump.
    while t_freefall <= 0:  # 当自由落体时间小于等于0时
        t_freefall = numeric_input("HOW MANY SECONDS", default=10)  # 通过用户输入获取自由落体时间，默认值为10
    return t_freefall  # 返回自由落体时间


def jump() -> float:
    """Simulate a jump and returns the altitude where the chute opened.

    The idea is to open the chute as late as possible -- but not too late.
    """
    v: float = 0  # 终端速度
    a: float = 0  # 加速度
    initial_altitude = int(9001 * random() + 1000)  # 初始高度为随机生成的9001到10000之间的整数

    v1 = get_terminal_velocity()  # 获取终端速度
    # Actual terminal velocity is +/-5% of v1.
    v = v1 * uniform(0.95, 1.05)  # 实际终端速度为v1的95%到105%
    a2 = get_acceleration()  # 获取加速度数据
    # 实际加速度为a2的正负5%
    a = a2 * uniform(0.95, 1.05)  # 计算实际加速度

    print(
        "\n"
        f"    ALTITUDE         = {initial_altitude} FT\n"  # 打印初始高度
        f"    TERM. VELOCITY   = {v1:.2f} FT/SEC +/-5%\n"  # 打印终端速度
        f"    ACCELERATION     = {a2:.2f} FT/SEC/SEC +/-5%\n"  # 打印加速度
        "SET THE TIMER FOR YOUR FREEFALL."  # 提示设置自由落体计时器
    )
    t_freefall = get_freefall_time()  # 获取自由落体时间
    print(
        "HERE WE GO.\n\n"
        "TIME (SEC)\tDIST TO FALL (FT)\n"
        "==========\t================="  # 打印表头
    )

    terminal_velocity_reached = False  # 初始化终端速度达到标志为False
    is_splat = False  # 初始化是否坠毁标志为False
    for i in range(9):
        # 将自由落体的时间分成8个间隔。
        t = i * (t_freefall / 8)
        # 根据第一个运动方程，v = v_0 + a * delta_t，其中初始速度 v_0 = 0，我们可以得到达到终端速度时的时间：delta_t = v / a。
        if t > v / a:
            if not terminal_velocity_reached:
                print(f"TERMINAL VELOCITY REACHED AT T PLUS {v / a:.2f} SECONDS.")
                terminal_velocity_reached = True
            # 达到终端速度后，位移由两部分组成：
            # 1. 达到终端速度之前的位移：
            #    根据第三个运动方程，v^2 = v_0^2 + 2 * a * d，其中 v_0 = 0，我们可以使用 d1 = v^2 / (2 * a) 得到位移。
            # 2. 超过达到终端速度后的位移：
            #    在这里，位移只是终端速度和达到终端速度后经过的时间的函数：d2 = v * (t - t_reached_term_vel)
            # 根据公式计算物体达到终端速度前的位移
            d1 = (v**2) / (2 * a)
            # 根据公式计算物体达到终端速度后的位移
            d2 = v * (t - (v / a))
            # 计算物体的高度
            altitude = initial_altitude - (d1 + d2)
            # 如果高度小于等于0，表示物体已经着地
            if altitude <= 0:
                # 计算物体落地所需的时间
                # 1. 达到终端速度前的时间：t1 = v / a
                # 2. 达到终端速度后的时间：altitude_remaining / v
                t1 = v / a
                t2 = (initial_altitude - d1) / v
                # 打印物体落地所需的总时间
                print_splat(t1 + t2)
                # 设置标志表示物体已经着地
                is_splat = True
                # 跳出循环
                break
        else:
            # 1. 达到终端速度前的位移
            #    根据第二个运动方程，
            #    d = v_0 * t + 0.5 * a * t^2，其中 v_0 = 0，我们可以得到
            #    位移使用 d1 = a / 2 * t^2
            d1 = (a / 2) * (t**2)
            altitude = initial_altitude - d1
            if altitude <= 0:
                # 如果高度小于等于0，表示物体已经落地，需要打印出落地时间
                # 使用第二个运动方程计算物体落地时间：
                # d = v_0 * t + 0.5 * a * t^2，其中 v_0 = 0，解出 t 为
                # t1 = sqrt(2 * d / a)。
                t1 = sqrt(2 * initial_altitude / a)
                print_splat(t1)
                is_splat = True
                break
        print(f"{t:.2f}\t\t{altitude:.1f}")

    if not is_splat:
        # 如果物体没有落地，表示降落伞已经打开
        print("CHUTE OPEN")
def pick_random_celestial_body() -> Tuple[str, float]:
    """Pick a random planet, the moon, or the sun with associated gravity."""
    # 从包含天体和对应重力的元组列表中随机选择一个元组
    return choice(
        [
            ("MERCURY", 12.2),
            ("VENUS", 28.3),
            ("EARTH", 32.16),
            ("THE MOON", 5.15),
            ("MARS", 12.5),
            ("JUPITER", 85.2),
            ("SATURN", 37.6),
            ("URANUS", 33.8),
            ("NEPTUNE", 39.6),
            ("THE SUN", 896.0),
        ]
    )
def jump_stats(previous_jumps, chute_altitude) -> Tuple[int, int]:
    """Compare altitude when chute opened with previous successful jumps.

    Return the number of previous jumps and the number of times
    the current jump is better.
    """
    # 计算之前成功跳伞的次数
    n_previous_jumps = len(previous_jumps)
    # 计算当前跳伞比之前更好的次数
    n_better = sum(1 for pj in previous_jumps if chute_altitude < pj)
    return n_previous_jumps, n_better


def print_splat(time_on_impact) -> None:
    """Parachute opened too late!"""
    # 打印跳伞时间和结果
    print(f"{time_on_impact:.2f}\t\tSPLAT")
    # 随机选择一条悼词
    print(
        choice(
            [
                "REQUIESCAT IN PACE.",
                "MAY THE ANGEL OF HEAVEN LEAD YOU INTO PARADISE.",
                "REST IN PEACE.",  # 添加注释：表示玩家失败时的提示信息
                "SON-OF-A-GUN.",  # 添加注释：表示玩家失败时的提示信息
                "#$%&&%!$",  # 添加注释：表示玩家失败时的提示信息
                "A KICK IN THE PANTS IS A BOOST IF YOU'RE HEADED RIGHT.",  # 添加注释：表示玩家失败时的提示信息
                "HMMM. SHOULD HAVE PICKED A SHORTER TIME.",  # 添加注释：表示玩家失败时的提示信息
                "MUTTER. MUTTER. MUTTER.",  # 添加注释：表示玩家失败时的提示信息
                "PUSHING UP DAISIES.",  # 添加注释：表示玩家失败时的提示信息
                "EASY COME, EASY GO.",  # 添加注释：表示玩家失败时的提示信息
            ]  # 添加注释：表示玩家失败时的提示信息
        )  # 添加注释：表示玩家失败时的提示信息


def print_results(n_previous_jumps, n_better) -> None:
    """Compare current jump to previous successful jumps."""  # 添加注释：打印当前跳跃与之前成功跳跃的比较结果
    k = n_previous_jumps  # 添加注释：将参数n_previous_jumps赋值给变量k
    k1 = n_better  # 添加注释：将参数n_better赋值给变量k1
    n_jumps = k + 1  # 添加注释：将k加1后的结果赋值给变量n_jumps
    if n_jumps <= 3:  # 添加注释：如果n_jumps小于等于3
        order = ["1ST", "2ND", "3RD"]  # 添加注释：创建一个包含"1ST", "2ND", "3RD"的列表赋值给变量order
        nth = order[n_jumps - 1]  # 从顺序列表中获取第n_jumps - 1个元素，赋值给nth
        print(f"AMAZING!!! NOT BAD FOR YOUR {nth} SUCCESSFUL JUMP!!!")  # 打印出成功跳跃的次数为nth的鼓励性消息
    elif k - k1 <= 0.1 * k:  # 如果k - k1小于等于总成功跳跃次数的10%
        print(
            f"WOW!  THAT'S SOME JUMPING.  OF THE {k} SUCCESSFUL JUMPS\n"
            f"BEFORE YOURS, ONLY {k - k1} OPENED THEIR CHUTES LOWER THAN\n"
            "YOU DID."
        )  # 打印出在你之前的成功跳跃中，有多少比例的人在你之前打开降落伞
    elif k - k1 <= 0.25 * k:  # 如果k - k1小于等于总成功跳跃次数的25%
        print(
            f"PRETTY GOOD!  {k} SUCCESSFUL JUMPS PRECEDED YOURS AND ONLY\n"
            f"{k - k1} OF THEM GOT LOWER THAN YOU DID BEFORE THEIR CHUTES\n"
            "OPENED."
        )  # 打印出在你之前的成功跳跃中，有多少比例的人在你之前打开降落伞
    elif k - k1 <= 0.5 * k:  # 如果k - k1小于等于总成功跳跃次数的50%
        print(
            f"NOT BAD.  THERE HAVE BEEN {k} SUCCESSFUL JUMPS BEFORE YOURS.\n"
            f"YOU WERE BEATEN OUT BY {k - k1} OF THEM."
        )  # 打印出在你之前的成功跳跃中，有多少比例的人在你之前打开降落伞
    elif k - k1 <= 0.75 * k:  # 如果k - k1小于等于总成功跳跃次数的75%
        print(
            f"CONSERVATIVE, AREN'T YOU?  YOU RANKED ONLY {k - k1} IN THE\n"
            f"{k} SUCCESSFUL JUMPS BEFORE YOURS."
        )
    # 如果跳伞成功的次数减去当前跳伞成功的次数小于等于总成功次数的90%，输出提示信息
    elif k - k1 <= 0.9 * k:
        print(
            "HUMPH!  DON'T YOU HAVE ANY SPORTING BLOOD?  THERE WERE\n"
            f"{k} SUCCESSFUL JUMPS BEFORE YOURS AND YOU CAME IN {k1} JUMPS\n"
            "BETTER THAN THE WORST.  SHAPE UP!!!"
        )
    # 如果以上条件都不满足，输出默认提示信息
    else:
        print(
            f"HEY!  YOU PULLED THE RIP CORD MUCH TOO SOON.  {k} SUCCESSFUL\n"
            f"JUMPS BEFORE YOURS AND YOU CAME IN NUMBER {k - k1}!"
            "  GET WITH IT!"
        )


def print_centered(msg: str) -> None:
    """Print centered text."""
```
在给定的代码中，第一部分是一个条件语句，根据不同的条件输出不同的提示信息。第二部分是一个函数定义，定义了一个打印居中文本的函数。
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)  # 根据消息长度计算需要添加的空格数，使得消息居中显示
    print(spaces + msg)  # 在屏幕上打印居中显示的消息


def print_header() -> None:
    print_centered("SPLAT")  # 调用print_centered函数打印居中显示的"SPLAT"
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 调用print_centered函数打印居中显示的"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print(
        "\n\n\n"
        "WELCOME TO 'SPLAT' -- THE GAME THAT SIMULATES A PARACHUTE\n"
        "JUMP.  TRY TO OPEN YOUR CHUTE AT THE LAST POSSIBLE\n"
        "MOMENT WITHOUT GOING SPLAT.\n\n"
    )  # 在屏幕上打印欢迎消息和游戏介绍


def main() -> None:
    print_header()  # 调用print_header函数打印游戏标题和介绍

    successful_jumps: List[float] = []  # 创建一个空的成功跳伞时间列表
    while True:  # 无限循环
        chute_altitude = jump()  # 调用 jump() 函数，获取跳伞高度
        if chute_altitude > 0:  # 如果跳伞高度大于0
            # 获取之前跳伞的统计数据（不包括当前跳伞）
            n_previous_jumps, n_better = jump_stats(successful_jumps, chute_altitude)
            successful_jumps.append(chute_altitude)  # 将当前跳伞高度添加到成功跳伞列表中
            print_results(n_previous_jumps, n_better)  # 打印统计结果
        else:  # 如果跳伞高度小于等于0
            # 落地了！
            print("I'LL GIVE YOU ANOTHER CHANCE.")  # 打印提示信息
        z = yes_no_input("DO YOU WANT TO PLAY AGAIN")  # 获取用户输入，是否想再玩一次
        if not z:  # 如果用户输入不是 yes
            z = yes_no_input("PLEASE")  # 再次询问用户是否想再玩一次
            if not z:  # 如果用户再次输入不是 yes
                print("SSSSSSSSSS.")  # 打印提示信息
                break  # 结束游戏循环

if __name__ == "__main__":
    main()  # 调用主函数开始游戏
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```