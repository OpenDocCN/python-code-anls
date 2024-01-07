# `basic-computer-games\12_Bombs_Away\python\bombs_away.py`

```

"""
Bombs away

Ported from BASIC to Python3 by Bernard Cooke (bernardcooke53)
Tested with Python 3.8.10, formatted with Black and type checked with mypy.
"""
import random
from typing import Iterable


def _stdin_choice(prompt: str, *, choices: Iterable[str]) -> str:
    ret = input(prompt)  # 获取用户输入的选择
    while ret not in choices:  # 如果选择不在给定的选项中
        print("TRY AGAIN...")  # 提示用户重新输入
        ret = input(prompt)  # 获取用户重新输入的选择
    return ret  # 返回用户的选择


def player_survived() -> None:
    print("YOU MADE IT THROUGH TREMENDOUS FLAK!!")  # 打印玩家幸存的消息


def player_death() -> None:
    print("* * * * BOOM * * * *")  # 打印玩家死亡的消息
    print("YOU HAVE BEEN SHOT DOWN.....")  # 打印玩家被击落的消息
    print("DEARLY BELOVED, WE ARE GATHERED HERE TODAY TO PAY OUR")  # 打印悼词
    print("LAST TRIBUTE...")  # 打印悼词


def mission_success() -> None:
    print(f"DIRECT HIT!!!! {int(100 * random.random())} KILLED.")  # 打印任务成功的消息
    print("MISSION SUCCESSFUL.")  # 打印任务成功的消息


def death_with_chance(p_death: float) -> bool:
    """
    Takes a float between 0 and 1 and returns a boolean
    if the player has survived (based on random chance)

    Returns True if death, False if survived
    """
    return p_death > random.random()  # 根据随机概率判断玩家是否幸存


# ... 省略部分注释 ...


def play_game() -> None:
    print("YOU ARE A PILOT IN A WORLD WAR II BOMBER.")  # 打印游戏开始的消息
    sides = {"1": play_italy, "2": play_allies, "3": play_japan, "4": play_germany}  # 定义不同阵营对应的游戏函数
    side = _stdin_choice(
        prompt="WHAT SIDE -- ITALY(1), ALLIES(2), JAPAN(3), GERMANY(4): ", choices=sides
    )  # 获取玩家选择的阵营
    return sides[side]()  # 执行玩家选择的阵营对应的游戏函数


if __name__ == "__main__":
    again = True
    while again:
        play_game()  # 执行游戏
        again = input("ANOTHER MISSION? (Y OR N): ").upper() == "Y"  # 获取玩家是否继续游戏的选择

```