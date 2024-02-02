# `basic-computer-games\52_Kinema\python\kinema.py`

```py
"""
KINEMA

A kinematics physics quiz.

Ported by Dave LeCompte
"""

import random

# We approximate gravity from 9.8 meters/second squared to 10, which
# is only off by about 2%. 10 is also a lot easier for people to use
# for mental math.

g = 10  # 设置重力加速度为10米/秒^2

# We only expect the student to get within this percentage of the
# correct answer. This isn't rocket science.

EXPECTED_ACCURACY_PERCENT = 15  # 设置预期答案的精度为15%


def do_quiz() -> None:
    print()
    print()
    num_questions_correct = 0

    # pick random initial velocity
    v0 = random.randint(5, 40)  # 随机选择初始速度
    print(f"A BALL IS THROWN UPWARDS AT {v0} METERS PER SECOND.")
    print()

    answer = v0**2 / (2 * g)  # 计算球的最大高度
    num_questions_correct += ask_player("HOW HIGH WILL IT GO (IN METERS)?", answer)

    answer = 2 * v0 / g  # 计算球返回所需的时间
    num_questions_correct += ask_player(
        "HOW LONG UNTIL IT RETURNS (IN SECONDS)?", answer
    )

    t = 1 + random.randint(0, 2 * v0) // g  # 随机选择时间
    answer = v0 - g * t  # 计算指定时间后的速度
    num_questions_correct += ask_player(
        f"WHAT WILL ITS VELOCITY BE AFTER {t} SECONDS?", answer
    )

    print()
    print(f"{num_questions_correct} right out of 3.")
    if num_questions_correct >= 2:
        print("  NOT BAD.")


def ask_player(question: str, answer) -> int:
    print(question)
    player_answer = float(input())  # 获取玩家的答案

    accuracy_frac = EXPECTED_ACCURACY_PERCENT / 100.0  # 计算精度的分数
    if abs((player_answer - answer) / answer) < accuracy_frac:  # 判断玩家答案是否在预期精度范围内
        print("CLOSE ENOUGH.")
        score = 1
    else:
        print("NOT EVEN CLOSE....")
        score = 0
    print(f"CORRECT ANSWER IS {answer}")  # 显示正确答案
    print()
    return score


def main() -> None:
    print(" " * 33 + "KINEMA")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

    while True:
        do_quiz()


if __name__ == "__main__":
    main()
```