# `basic-computer-games\41_Guess\python\guess.py`

```py
"""
Guess

From: Basic Computer Games (1978)

 "In program Guess, the computer  chooses a random
  integer between 0 and any limit and any limit you
  set. You must then try to guess the number the
  computer has choosen using the clues provideed by
  the computer.
   You should be able to guess the number in one less
  than the number of digits needed to  represent the
  number in binary notation - i.e. in base 2. This ought
  to give you a clue as to the optimum search technique.
   Guess converted from the original program in FOCAL
  which appeared in the book "Computers in the Classroom"
  by Walt Koetke of Lexington High School, Lexington,
  Massaschusetts.
"""

# Altough the introduction says that the computer chooses
# a number between 0 and any limit, it actually chooses
# a number between 1 and any limit. This due to the fact that
# for computing the number of digits the limit has in binary
# representation, it has to use log.

from math import log  # 导入 log 函数
from random import random  # 导入 random 函数
from typing import Tuple  # 导入 Tuple 类型


def insert_whitespaces() -> None:  # 定义函数，不返回任何值
    print("\n\n\n\n\n")  # 打印空行


def limit_set() -> Tuple[int, int]:  # 定义函数，返回元组类型
    print("                   Guess")  # 打印标题
    print("Creative Computing  Morristown, New Jersey")  # 打印作者信息
    print("\n\n\n")  # 打印空行
    print("This is a number guessing game. I'll think")  # 打印游戏介绍
    print("of a number between 1 and any limit you want.\n")
    print("Then you have to guess what it is\n")
    print("What limit do you want?")  # 提示用户输入限制

    limit = int(input())  # 获取用户输入的限制

    while limit <= 0:  # 当限制小于等于0时
        print("Please insert a number greater or equal to 1")  # 提示用户输入大于等于1的数字
        limit = int(input())  # 获取用户输入的限制

    # limit_goal = Number of digits "limit" in binary has
    limit_goal = int((log(limit) / log(2)) + 1)  # 计算限制在二进制表示中的位数

    return limit, limit_goal  # 返回限制和限制在二进制表示中的位数


def main() -> None:  # 定义函数，不返回任何值
    limit, limit_goal = limit_set()  # 调用函数获取限制和限制在二进制表示中的位数
    # 无限循环，直到猜对为止
    while True:
        # 猜测次数初始化为1
        guess_count = 1
        # 是否还在猜测的标志
        still_guessing = True
        # 是否赢了的标志
        won = False
        # 生成一个1到limit之间的随机数作为猜测的数字
        my_guess = int(limit * random() + 1)

        # 打印提示信息，告诉用户要猜的数字范围
        print(f"I'm thinking of a number between 1 and {limit}")
        print("Now you try to guess what it is.")

        # 进入猜测循环
        while still_guessing:
            # 获取用户输入的猜测数字
            n = int(input())

            # 如果用户输入的是负数，则退出猜测循环
            if n < 0:
                break

            # 调用insert_whitespaces函数
            insert_whitespaces()
            # 判断用户猜测的数字与随机数的大小关系，并给出相应提示
            if n < my_guess:
                print("Too low. Try a bigger answer")
                guess_count += 1
            elif n > my_guess:
                print("Too high. Try a smaller answer")
                guess_count += 1
            else:
                # 猜对了，打印提示信息，更新标志位
                print(f"That's it! You got it in {guess_count} tries")
                won = True
                still_guessing = False

        # 判断是否赢了
        if won:
            # 根据猜测次数给出不同的评价
            if guess_count < limit_goal:
                print("Very good.")
            elif guess_count == limit_goal:
                print("Good.")
            else:
                print(f"You should have been able to get it in only {limit_goal}")
            # 调用insert_whitespaces函数
            insert_whitespaces()
        else:
            # 调用limit_set函数，更新limit和limit_goal的值
            insert_whitespaces()
            limit, limit_goal = limit_set()
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```