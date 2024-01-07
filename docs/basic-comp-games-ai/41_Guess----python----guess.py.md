# `basic-computer-games\41_Guess\python\guess.py`

```

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

from math import log  # 导入log函数
from random import random  # 导入random函数
from typing import Tuple  # 导入Tuple类型


def insert_whitespaces() -> None:  # 定义函数，打印空行
    print("\n\n\n\n\n")


def limit_set() -> Tuple[int, int]:  # 定义函数，设置限制
    print("                   Guess")  # 打印标题
    print("Creative Computing  Morristown, New Jersey")  # 打印作者信息
    print("\n\n\n")  # 打印空行
    print("This is a number guessing game. I'll think")  # 打印游戏介绍
    print("of a number between 1 and any limit you want.\n")
    print("Then you have to guess what it is\n")
    print("What limit do you want?")

    limit = int(input())  # 获取用户输入的限制

    while limit <= 0:  # 如果限制小于等于0
        print("Please insert a number greater or equal to 1")  # 提示用户输入大于等于1的数字
        limit = int(input())  # 获取用户输入的限制

    # limit_goal = Number of digits "limit" in binary has
    limit_goal = int((log(limit) / log(2)) + 1)  # 计算限制在二进制表示中的位数

    return limit, limit_goal  # 返回限制和限制在二进制表示中的位数


def main() -> None:  # 定义主函数
    limit, limit_goal = limit_set()  # 获取限制和限制在二进制表示中的位数
    while True:  # 无限循环
        guess_count = 1  # 猜测次数初始化为1
        still_guessing = True  # 是否继续猜测的标志初始化为True
        won = False  # 是否猜中的标志初始化为False
        my_guess = int(limit * random() + 1)  # 生成计算机的猜测数字

        print(f"I'm thinking of a number between 1 and {limit}")  # 打印计算机思考的范围
        print("Now you try to guess what it is.")  # 提示用户猜测数字

        while still_guessing:  # 循环直到不再猜测
            n = int(input())  # 获取用户输入的猜测数字

            if n < 0:  # 如果猜测数字小于0
                break  # 退出循环

            insert_whitespaces()  # 调用打印空行的函数
            if n < my_guess:  # 如果猜测数字小于计算机的猜测数字
                print("Too low. Try a bigger answer")  # 提示猜测数字过小
                guess_count += 1  # 猜测次数加1
            elif n > my_guess:  # 如果猜测数字大于计算机的猜测数字
                print("Too high. Try a smaller answer")  # 提示猜测数字过大
                guess_count += 1  # 猜测次数加1
            else:  # 如果猜测数字等于计算机的猜测数字
                print(f"That's it! You got it in {guess_count} tries")  # 提示猜中了，并显示猜测次数
                won = True  # 设置猜中标志为True
                still_guessing = False  # 设置继续猜测的标志为False

        if won:  # 如果猜中了
            if guess_count < limit_goal:  # 如果猜测次数小于限制在二进制表示中的位数
                print("Very good.")  # 提示猜测次数很好
            elif guess_count == limit_goal:  # 如果猜测次数等于限制在二进制表示中的位数
                print("Good.")  # 提示猜测次数还行
            else:  # 如果猜测次数大于限制在二进制表示中的位数
                print(f"You should have been able to get it in only {limit_goal}")  # 提示应该能在指定次数内猜中
            insert_whitespaces()  # 调用打印空行的函数
        else:  # 如果没有猜中
            insert_whitespaces()  # 调用打印空行的函数
            limit, limit_goal = limit_set()  # 重新设置限制和限制在二进制表示中的位数


if __name__ == "__main__":  # 如果是主程序入口
    main()  # 调用主函数

```