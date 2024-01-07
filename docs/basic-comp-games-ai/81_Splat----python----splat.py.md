# `basic-computer-games\81_Splat\python\splat.py`

```

# 导入所需的库
from math import sqrt
from random import choice, random, uniform
from typing import List, Tuple

# 页面宽度
PAGE_WIDTH = 72

# 询问用户输入一个数字值
def numeric_input(question, default=0) -> float:
    """Ask user for a numeric value."""
    while True:
        answer_str = input(f"{question} [{default}]: ").strip() or default
        try:
            return float(answer_str)
        except ValueError:
            pass

# 询问用户是/否问题，并返回True或False
def yes_no_input(question: str, default="YES") -> bool:
    """Ask user a yes/no question and returns True if yes, otherwise False."""
    answer = input(f"{question} (YES OR NO) [{default}]: ").strip() or default
    while answer.lower() not in ["n", "no", "y", "yes"]:
        answer = input(f"YES OR NO [{default}]: ").strip() or default
    return answer.lower() in ["y", "yes"]

# 获取终端速度
def get_terminal_velocity() -> float:
    """Terminal velocity by user or picked by computer."""
    if yes_no_input("SELECT YOUR OWN TERMINAL VELOCITY", default="NO"):
        v1 = numeric_input("WHAT TERMINAL VELOCITY (MI/HR)", default=100)
    else:
        # 计算机随机选择0-1000的终端速度
        v1 = int(1000 * random())
        print(f"OK.  TERMINAL VELOCITY = {v1} MI/HR")

    # 将英里/小时转换为英尺/秒
    return v1 * (5280 / 3600)

# 获取加速度
def get_acceleration() -> float:
    """Acceleration due to gravity by user or picked by computer."""
    if yes_no_input("WANT TO SELECT ACCELERATION DUE TO GRAVITY", default="NO"):
        a2 = numeric_input("WHAT ACCELERATION (FT/SEC/SEC)", default=32.16)
    else:
        body, a2 = pick_random_celestial_body()
        print(f"FINE. YOU'RE ON {body}. ACCELERATION={a2} FT/SEC/SEC.")
    return a2

# 获取自由落体时间
def get_freefall_time() -> float:
    """User-guessed freefall time."""
    t_freefall: float = 0
    # 不能处理零或负的自由落体时间
    while t_freefall <= 0:
        t_freefall = numeric_input("HOW MANY SECONDS", default=10)
    return t_freefall

# 随机选择天体和对应的重力
def pick_random_celestial_body() -> Tuple[str, float]:
    """Pick a random planet, the moon, or the sun with associated gravity."""
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

# 比较开伞时的高度与先前成功跳伞的高度
def jump_stats(previous_jumps, chute_altitude) -> Tuple[int, int]:
    """Compare altitude when chute opened with previous successful jumps.

    Return the number of previous jumps and the number of times
    the current jump is better.
    """
    n_previous_jumps = len(previous_jumps)
    n_better = sum(1 for pj in previous_jumps if chute_altitude < pj)
    return n_previous_jumps, n_better

# 打印跳伞失败信息
def print_splat(time_on_impact) -> None:
    """Parachute opened too late!"""
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

# 打印跳伞结果
def print_results(n_previous_jumps, n_better) -> None:
    """Compare current jump to previous successful jumps."""
    k = n_previous_jumps
    k1 = n_better
    n_jumps = k + 1
    if n_jumps <= 3:
        order = ["1ST", "2ND", "3RD"]
        nth = order[n_jumps - 1]
        print(f"AMAZING!!! NOT BAD FOR YOUR {nth} SUCCESSFUL JUMP!!!")
    elif k - k1 <= 0.1 * k:
        print(
            f"WOW!  THAT'S SOME JUMPING.  OF THE {k} SUCCESSFUL JUMPS\n"
            f"BEFORE YOURS, ONLY {k - k1} OPENED THEIR CHUTES LOWER THAN\n"
            "YOU DID."
        )
    # 更多的条件判断...

# 打印居中文本
def print_centered(msg: str) -> None:
    """Print centered text."""
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)

# 打印游戏标题
def print_header() -> None:
    print_centered("SPLAT")
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print(
        "\n\n\n"
        "WELCOME TO 'SPLAT' -- THE GAME THAT SIMULATES A PARACHUTE\n"
        "JUMP.  TRY TO OPEN YOUR CHUTE AT THE LAST POSSIBLE\n"
        "MOMENT WITHOUT GOING SPLAT.\n\n"
    )

# 主函数
def main() -> None:
    print_header()

    successful_jumps: List[float] = []
    while True:
        chute_altitude = jump()
        if chute_altitude > 0:
            # We want the statistics on previous jumps (i.e. not including the
            # current jump.)
            n_previous_jumps, n_better = jump_stats(successful_jumps, chute_altitude)
            successful_jumps.append(chute_altitude)
            print_results(n_previous_jumps, n_better)
        else:
            # Splat!
            print("I'LL GIVE YOU ANOTHER CHANCE.")
        z = yes_no_input("DO YOU WANT TO PLAY AGAIN")
        if not z:
            z = yes_no_input("PLEASE")
            if not z:
                print("SSSSSSSSSS.")
                break

# 如果是主程序，则执行主函数
if __name__ == "__main__":
    main()

```