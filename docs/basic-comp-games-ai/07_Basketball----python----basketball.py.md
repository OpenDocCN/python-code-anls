# `basic-computer-games\07_Basketball\python\basketball.py`

```

"""
篮球类是一个电脑游戏，允许你扮演达特茅斯学院的队长和组织者
该游戏使用设定的概率来模拟每次进攻的结果
你可以选择投篮类型以及防守阵型
"""

import random
from typing import List, Literal, Optional


def print_intro() -> None:
    # 打印游戏介绍
    print("\t\t\t Basketball")
    print("\t Creative Computing  Morristown, New Jersey\n\n\n")
    print("This is Dartmouth College basketball. ")
    print("Υou will be Dartmouth captain and playmaker.")
    print("Call shots as follows:")
    print(
        "1. Long (30ft.) Jump Shot; "
        "2. Short (15 ft.) Jump Shot; "
        "3. Lay up; 4. Set Shot"
    )
    print("Both teams will use the same defense. Call Defense as follows:")
    print("6. Press; 6.5 Man-to-Man; 7. Zone; 7.5 None.")
    print("To change defense, just type 0 as your next shot.")
    print("Your starting defense will be? ", end="")


def get_defense_choice(defense_choices: List[float]) -> float:
    """获取防守选择"""
    try:
        defense = float(input())
    except ValueError:
        defense = None

    # 如果输入不是有效的防守选择，则重新输入
    while defense not in defense_choices:
        print("Your new defensive allignment is? ", end="")
        try:
            defense = float(input())
        except ValueError:
            continue
    assert isinstance(defense, float)
    return defense


def get_dartmouth_ball_choice(shot_choices: List[Literal[0, 1, 2, 3, 4]]) -> int:
    # 获取达特茅斯队的投篮选择
    print("Your shot? ", end="")
    shot = None
    try:
        shot = int(input())
    except ValueError:
        shot = None

    while shot not in shot_choices:
        print("Incorrect answer. Retype it. Your shot? ", end="")
        try:
            shot = int(input())
        except Exception:
            continue
    assert isinstance(shot, int)
    return shot


def get_opponents_name() -> str:
    """获取对手的名字"""
    print("\nChoose your opponent? ", end="")
    return input()


if __name__ == "__main__":
    Basketball()

```