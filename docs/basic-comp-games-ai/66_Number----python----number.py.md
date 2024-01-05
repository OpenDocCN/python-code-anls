# `66_Number\python\number.py`

```
"""
NUMBER

A number guessing (gambling) game.

Ported by Dave LeCompte
"""

import random  # 导入 random 模块


def print_instructions() -> None:  # 定义函数 print_instructions，无返回值
    print("YOU HAVE 100 POINTS.  BY GUESSING NUMBERS FROM 1 TO 5, YOU")  # 打印提示信息
    print("CAN GAIN OR LOSE POINTS DEPENDING UPON HOW CLOSE YOU GET TO")  # 打印提示信息
    print("A RANDOM NUMBER SELECTED BY THE COMPUTER.")  # 打印提示信息
    print()  # 打印空行
    print("YOU OCCASIONALLY WILL GET A JACKPOT WHICH WILL DOUBLE(!)")  # 打印提示信息
    print("YOUR POINT COUNT.  YOU WIN WHEN YOU GET 500 POINTS.")  # 打印提示信息
    print()  # 打印空行
def fnr() -> int:  # 定义一个函数fnr，返回一个整数
    return random.randint(1, 5)  # 返回一个1到5之间的随机整数


def main() -> None:  # 定义一个主函数main，不返回任何值
    print(" " * 33 + "NUMBER")  # 打印空格乘以33再加上"NUMBER"
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 打印空格乘以15再加上"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"

    print_instructions()  # 调用打印指令的函数

    points: float = 100  # 初始化一个浮点数变量points为100

    while points <= 500:  # 当points小于等于500时执行循环
        print("GUESS A NUMBER FROM 1 TO 5")  # 打印提示信息
        guess = int(input())  # 获取用户输入的整数并赋值给变量guess

        if (guess < 1) or (guess > 5):  # 如果guess小于1或者大于5
            continue  # 继续下一次循环
        r = fnr()  # 从函数 fnr() 中获取一个随机数并赋值给变量 r
        s = fnr()  # 从函数 fnr() 中获取一个随机数并赋值给变量 s
        t = fnr()  # 从函数 fnr() 中获取一个随机数并赋值给变量 t
        u = fnr()  # 从函数 fnr() 中获取一个随机数并赋值给变量 u
        v = fnr()  # 从函数 fnr() 中获取一个随机数并赋值给变量 v

        if guess == r:  # 如果猜测的数字等于 r
            # lose 5  # 失去 5 分
            points -= 5  # 分数减去 5
        elif guess == s:  # 如果猜测的数字等于 s
            # gain 5  # 获得 5 分
            points += 5  # 分数加上 5
        elif guess == t:  # 如果猜测的数字等于 t
            # double!  # 翻倍！
            points += points  # 分数加上自身，相当于翻倍
            print("YOU HIT THE JACKPOT!!!")  # 打印“你中了大奖！”
        elif guess == u:  # 如果猜测的数字等于 u
            # gain 1  # 获得 1 分
            points += 1  # 分数加上 1
        elif guess == v:  # 如果猜测的数字等于 v
# lose half
# 减去一半的分数
points = points - (points * 0.5)

# 打印剩余的分数
print(f"YOU HAVE {points} POINTS.")
print()

# 打印最终获得的分数
print(f"!!!!YOU WIN!!!! WITH {points} POINTS.")


if __name__ == "__main__":
    main()
```