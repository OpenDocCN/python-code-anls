# `basic-computer-games\81_Splat\python\splat.py`

```
# 导入所需的数学函数和随机函数
from math import sqrt
from random import choice, random, uniform
# 导入类型提示
from typing import List, Tuple

# 页面宽度常量
PAGE_WIDTH = 72

# 询问用户输入一个数值
def numeric_input(question, default=0) -> float:
    """Ask user for a numeric value."""
    while True:
        answer_str = input(f"{question} [{default}]: ").strip() or default
        try:
            return float(answer_str)
        except ValueError:
            pass

# 询问用户是或否的问题，返回True或False
def yes_no_input(question: str, default="YES") -> bool:
    """Ask user a yes/no question and returns True if yes, otherwise False."""
    answer = input(f"{question} (YES OR NO) [{default}]: ").strip() or default
    while answer.lower() not in ["n", "no", "y", "yes"]:
        answer = input(f"YES OR NO [{default}]: ").strip() or default
    return answer.lower() in ["y", "yes"]

# 获取终端速度，用户输入或计算机选择
def get_terminal_velocity() -> float:
    """Terminal velocity by user or picked by computer."""
    # 如果用户选择自定义终端速度
    if yes_no_input("SELECT YOUR OWN TERMINAL VELOCITY", default="NO"):
        # 用户输入终端速度，单位为英里/小时
        v1 = numeric_input("WHAT TERMINAL VELOCITY (MI/HR)", default=100)
    else:
        # 计算机随机选择0-1000的终端速度
        v1 = int(1000 * random())
        # 打印终端速度
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
        # 计算机随机选择天体并返回重力加速度
        body, a2 = pick_random_celestial_body()
        print(f"FINE. YOU'RE ON {body}. ACCELERATION={a2} FT/SEC/SEC.")
    return a2


def get_freefall_time() -> float:
    """User-guessed freefall time.

    The idea of the game is to pick a freefall time, given initial
    altitude, terminal velocity and acceleration, so the parachute
    as close to the ground as possible without going splat.
    """
    # 用户输入自由落体时间
    t_freefall: float = 0
    # 处理零或负的自由落体时间
    while t_freefall <= 0:
        t_freefall = numeric_input("HOW MANY SECONDS", default=10)
    return t_freefall


def jump() -> float:
    """Simulate a jump and returns the altitude where the chute opened.

    The idea is to open the chute as late as possible -- but not too late.
    """
    # 终端速度和加速度的初始化
    v: float = 0  # Terminal velocity.
    a: float = 0  # Acceleration.
    # 初始高度
    initial_altitude = int(9001 * random() + 1000)

    # 获取终端速度
    v1 = get_terminal_velocity()
    # 实际终端速度为v1的正负5%
    v = v1 * uniform(0.95, 1.05)

    # 获取加速度
    a2 = get_acceleration()
    # 实际加速度为a2的正负5%
    a = a2 * uniform(0.95, 1.05)

    print(
        "\n"
        f"    ALTITUDE         = {initial_altitude} FT\n"
        f"    TERM. VELOCITY   = {v1:.2f} FT/SEC +/-5%\n"
        f"    ACCELERATION     = {a2:.2f} FT/SEC/SEC +/-5%\n"
        "SET THE TIMER FOR YOUR FREEFALL."
    )
    # 获取自由落体时间
    t_freefall = get_freefall_time()
    print(
        "HERE WE GO.\n\n"
        "TIME (SEC)\tDIST TO FALL (FT)\n"
        "==========\t================="
    )

    # 初始化变量
    terminal_velocity_reached = False
    is_splat = False
    if not is_splat:
        print("CHUTE OPEN")
    return altitude
# 从给定的星球、月球或太阳中随机选择一个，并返回其名称和对应的重力加速度
def pick_random_celestial_body() -> Tuple[str, float]:
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


# 比较降落伞打开时的高度与先前成功跳伞的次数
# 返回先前跳伞次数和当前跳伞比先前跳伞更好的次数
def jump_stats(previous_jumps, chute_altitude) -> Tuple[int, int]:
    n_previous_jumps = len(previous_jumps)
    n_better = sum(1 for pj in previous_jumps if chute_altitude < pj)
    return n_previous_jumps, n_better


# 打印“SPLAT”，表示降落伞打开太晚导致坠落
def print_splat(time_on_impact) -> None:
    print(f"{time_on_impact:.2f}\t\tSPLAT")
    print(
        choice(
            [
                "REQUIESCAT IN PACE.",
                "MAY THE ANGEL OF HEAVEN LEAD YOU INTO PARADISE.",
                "REST IN PEACE.",
                "SON-OF-A-GUN.",
                "#$%&&%!$",
                "A KICK IN THE PANTS IS A BOOST IF YOU'RE HEADED RIGHT.",
                "HMMM. SHOULD HAVE PICKED A SHORTER TIME.",
                "MUTTER. MUTTER. MUTTER.",
                "PUSHING UP DAISIES.",
                "EASY COME, EASY GO.",
            ]
        )
    )


# 比较当前跳伞与先前成功跳伞的次数
# 打印当前跳伞是第几次成功跳伞
def print_results(n_previous_jumps, n_better) -> None:
    k = n_previous_jumps
    k1 = n_better
    n_jumps = k + 1
    if n_jumps <= 3:
        order = ["1ST", "2ND", "3RD"]
        nth = order[n_jumps - 1]
        print(f"AMAZING!!! NOT BAD FOR YOUR {nth} SUCCESSFUL JUMP!!!")
    # 如果跳伞高度差在总高度的10%以内
    elif k - k1 <= 0.1 * k:
        # 打印成功跳伞的数量和比你跳伞高度低的数量
        print(
            f"WOW!  THAT'S SOME JUMPING.  OF THE {k} SUCCESSFUL JUMPS\n"
            f"BEFORE YOURS, ONLY {k - k1} OPENED THEIR CHUTES LOWER THAN\n"
            "YOU DID."
        )
    # 如果跳伞高度差在总高度的25%以内
    elif k - k1 <= 0.25 * k:
        # 打印成功跳伞的数量和比你跳伞高度低的数量
        print(
            f"PRETTY GOOD!  {k} SUCCESSFUL JUMPS PRECEDED YOURS AND ONLY\n"
            f"{k - k1} OF THEM GOT LOWER THAN YOU DID BEFORE THEIR CHUTES\n"
            "OPENED."
        )
    # 如果跳伞高度差在总高度的50%以内
    elif k - k1 <= 0.5 * k:
        # 打印成功跳伞的数量和比你跳伞高度低的数量
        print(
            f"NOT BAD.  THERE HAVE BEEN {k} SUCCESSFUL JUMPS BEFORE YOURS.\n"
            f"YOU WERE BEATEN OUT BY {k - k1} OF THEM."
        )
    # 如果跳伞高度差在总高度的75%以内
    elif k - k1 <= 0.75 * k:
        # 打印成功跳伞的数量和比你跳伞高度低的数量
        print(
            f"CONSERVATIVE, AREN'T YOU?  YOU RANKED ONLY {k - k1} IN THE\n"
            f"{k} SUCCESSFUL JUMPS BEFORE YOURS."
        )
    # 如果跳伞高度差在总高度的90%以内
    elif k - k1 <= 0.9 * k:
        # 打印成功跳伞的数量和比你跳伞高度低的数量
        print(
            "HUMPH!  DON'T YOU HAVE ANY SPORTING BLOOD?  THERE WERE\n"
            f"{k} SUCCESSFUL JUMPS BEFORE YOURS AND YOU CAME IN {k1} JUMPS\n"
            "BETTER THAN THE WORST.  SHAPE UP!!!"
        )
    # 如果跳伞高度差超过总高度的90%
    else:
        # 打印成功跳伞的数量和比你跳伞高度低的数量
        print(
            f"HEY!  YOU PULLED THE RIP CORD MUCH TOO SOON.  {k} SUCCESSFUL\n"
            f"JUMPS BEFORE YOURS AND YOU CAME IN NUMBER {k - k1}!"
            "  GET WITH IT!"
        )
# 定义一个函数，用于打印居中文本
def print_centered(msg: str) -> None:
    """Print centered text."""
    # 计算需要添加的空格数，使得文本居中
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    # 打印居中文本
    print(spaces + msg)


# 定义一个函数，用于打印标题
def print_header() -> None:
    # 打印居中的标题
    print_centered("SPLAT")
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    # 打印游戏介绍
    print(
        "\n\n\n"
        "WELCOME TO 'SPLAT' -- THE GAME THAT SIMULATES A PARACHUTE\n"
        "JUMP.  TRY TO OPEN YOUR CHUTE AT THE LAST POSSIBLE\n"
        "MOMENT WITHOUT GOING SPLAT.\n\n"
    )


# 定义主函数
def main() -> None:
    # 打印游戏标题
    print_header()

    # 初始化成功跳伞的高度列表
    successful_jumps: List[float] = []
    # 循环进行游戏
    while True:
        # 进行跳伞，获取跳伞高度
        chute_altitude = jump()
        if chute_altitude > 0:
            # 统计之前跳伞的次数和比当前跳伞更成功的次数
            n_previous_jumps, n_better = jump_stats(successful_jumps, chute_altitude)
            # 将当前跳伞高度添加到成功跳伞的高度列表中
            successful_jumps.append(chute_altitude)
            # 打印跳伞结果
            print_results(n_previous_jumps, n_better)
        else:
            # 跳伞失败
            print("I'LL GIVE YOU ANOTHER CHANCE.")
        # 询问是否继续游戏
        z = yes_no_input("DO YOU WANT TO PLAY AGAIN")
        if not z:
            z = yes_no_input("PLEASE")
            if not z:
                print("SSSSSSSSSS.")
                # 结束游戏
                break


# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```