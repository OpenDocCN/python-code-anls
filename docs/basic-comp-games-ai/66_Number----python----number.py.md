# `basic-computer-games\66_Number\python\number.py`

```

"""
NUMBER

A number guessing (gambling) game.

Ported by Dave LeCompte
"""

import random


def print_instructions() -> None:
    # 打印游戏说明
    print("YOU HAVE 100 POINTS.  BY GUESSING NUMBERS FROM 1 TO 5, YOU")
    print("CAN GAIN OR LOSE POINTS DEPENDING UPON HOW CLOSE YOU GET TO")
    print("A RANDOM NUMBER SELECTED BY THE COMPUTER.")
    print()
    print("YOU OCCASIONALLY WILL GET A JACKPOT WHICH WILL DOUBLE(!)")
    print("YOUR POINT COUNT.  YOU WIN WHEN YOU GET 500 POINTS.")
    print()


def fnr() -> int:
    # 生成一个1到5的随机整数
    return random.randint(1, 5)


def main() -> None:
    # 打印游戏标题
    print(" " * 33 + "NUMBER")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")

    # 打印游戏说明
    print_instructions()

    # 初始化玩家的初始点数
    points: float = 100

    # 当玩家的点数小于等于500时，进行游戏循环
    while points <= 500:
        print("GUESS A NUMBER FROM 1 TO 5")
        # 玩家输入猜测的数字
        guess = int(input())

        # 如果玩家输入的数字不在1到5之间，则重新进行循环
        if (guess < 1) or (guess > 5):
            continue

        # 生成5个随机数
        r = fnr()
        s = fnr()
        t = fnr()
        u = fnr()
        v = fnr()

        # 根据玩家猜测的数字和随机数的比较，更新玩家的点数
        if guess == r:
            # 失去5点
            points -= 5
        elif guess == s:
            # 赢得5点
            points += 5
        elif guess == t:
            # 翻倍
            points += points
            print("YOU HIT THE JACKPOT!!!")
        elif guess == u:
            # 赢得1点
            points += 1
        elif guess == v:
            # 失去一半的点数
            points = points - (points * 0.5)

        # 打印玩家当前的点数
        print(f"YOU HAVE {points} POINTS.")
        print()
    # 打印玩家最终的点数
    print(f"!!!!YOU WIN!!!! WITH {points} POINTS.")


if __name__ == "__main__":
    main()

```