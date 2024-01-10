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
    # 生成一个1到5之间的随机整数
    return random.randint(1, 5)


def main() -> None:
    print(" " * 33 + "NUMBER")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")

    print_instructions()

    points: float = 100

    while points <= 500:
        print("GUESS A NUMBER FROM 1 TO 5")
        guess = int(input())

        if (guess < 1) or (guess > 5):
            continue

        r = fnr()
        s = fnr()
        t = fnr()
        u = fnr()
        v = fnr()

        if guess == r:
            # 失去5分
            points -= 5
        elif guess == s:
            # 赢得5分
            points += 5
        elif guess == t:
            # 翻倍！
            points += points
            print("YOU HIT THE JACKPOT!!!")
        elif guess == u:
            # 赢得1分
            points += 1
        elif guess == v:
            # 失去一半
            points = points - (points * 0.5)

        print(f"YOU HAVE {points} POINTS.")
        print()
    print(f"!!!!YOU WIN!!!! WITH {points} POINTS.")


if __name__ == "__main__":
    main()
```