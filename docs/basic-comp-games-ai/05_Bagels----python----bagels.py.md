# `basic-computer-games\05_Bagels\python\bagels.py`

```
"""
Bagels

From: BASIC Computer Games (1978)
      Edited by David H. Ahl

"In this game, the computer picks a 3-digit secret number using
 the digits 0 to 9 and you attempt to guess what it is.  You are
 allowed up to twenty guesses.  No digit is repeated.  After
 each guess the computer will give you clues about your guess
 as follows:

 PICO     One digit is correct, but in the wrong place
 FERMI    One digit is in the correct place
 BAGELS   No digit is correct

"You will learn to draw inferences from the clues and, with
 practice, you'll learn to improve your score.  There are several
 good strategies for playing Bagels.  After you have found a good
 strategy, see if you can improve it.  Or try a different strategy
 altogether and see if it is any better.  While the program allows
 up to twenty guesses, if you use a good strategy it should not
 take more than eight guesses to get any number.

"The original authors of this program are D. Resek and P. Rowe of
 the Lawrence Hall of Science, Berkeley, California."


Python port by Jeff Jetton, 2019
"""


import random
from typing import List

MAX_GUESSES = 20


def print_rules() -> None:
    # 打印游戏规则
    print("\nI am thinking of a three-digit number.  Try to guess")
    print("my number and I will give you clues as follows:")
    print("   PICO   - One digit correct but in the wrong position")
    print("   FERMI  - One digit correct and in the right position")
    print("   BAGELS - No digits correct")


def pick_number() -> List[str]:
    # 选择一个三位数作为秘密数字
    # 注意：这里返回的是一个包含每个数字的单独字符串的列表，而不是一个整数或字符串
    numbers = list(range(10))
    random.shuffle(numbers)
    num = numbers[0:3]
    num_str = [str(i) for i in num]
    return num_str


def get_valid_guess(guesses: int) -> str:
    # 获取有效的猜测
    valid = False
    # 当猜测不正确时，进入循环
    while not valid:
        # 获取用户输入的猜测值，并显示猜测次数
        guess = input(f"Guess # {guesses}     ? ")
        # 去除输入值两端的空格
        guess = guess.strip()
        # 猜测值必须是三个字符
        if len(guess) == 3:
            # 并且它们应该是数字
            if guess.isnumeric():
                # 并且这些数字必须是唯一的
                if len(set(guess)) == 3:
                    # 猜测值有效，退出循环
                    valid = True
                else:
                    # 打印错误消息，提示用户猜测值中不能有相同的数字
                    print("Oh, I forgot to tell you that the number I have in mind")
                    print("has no two digits the same.")
            else:
                # 打印错误消息，提示用户输入必须是数字
                print("What?")
        else:
            # 打印错误消息，提示用户猜测值必须是三位数
            print("Try guessing a three-digit number.")

    # 返回有效的猜测值
    return guess
def build_result_string(num: List[str], guess: str) -> str:
    result = ""

    # Correct digits in wrong place
    for i in range(2):
        # 检查数字是否在错误的位置
        if num[i] == guess[i + 1]:
            result += "PICO "
        if num[i + 1] == guess[i]:
            result += "PICO "
    if num[0] == guess[2]:
        result += "PICO "
    if num[2] == guess[0]:
        result += "PICO "

    # Correct digits in right place
    for i in range(3):
        # 检查数字是否在正确的位置
        if num[i] == guess[i]:
            result += "FERMI "

    # Nothing right?
    if result == "":
        result = "BAGELS"

    return result


def main() -> None:
    # Intro text
    print("\n                Bagels")
    print("Creative Computing  Morristown, New Jersey\n\n")

    # Anything other than N* will show the rules
    response = input("Would you like the rules (Yes or No)? ")
    if len(response) > 0:
        if response.upper()[0] != "N":
            print_rules()
    else:
        print_rules()

    games_won = 0
    still_running = True
    while still_running:

        # New round
        num = pick_number()
        num_str = "".join(num)
        guesses = 1

        print("\nO.K.  I have a number in mind.")
        guessing = True
        while guessing:

            guess = get_valid_guess(guesses)

            if guess == num_str:
                print("You got it!!!\n")
                games_won += 1
                guessing = False
            else:
                print(build_result_string(num, guess))
                guesses += 1
                if guesses > MAX_GUESSES:
                    print("Oh well")
                    print(f"That's {MAX_GUESSES} guesses.  My number was {num_str}")
                    guessing = False

        valid_response = False
        while not valid_response:
            response = input("Play again (Yes or No)? ")
            if len(response) > 0:
                valid_response = True
                if response.upper()[0] != "Y":
                    still_running = False
    # 如果赢得的游戏数大于0，则打印出获得的积分加成信息
    if games_won > 0:
        print(f"\nA {games_won} point Bagels buff!!")

    # 打印结束语
    print("Hope you had fun.  Bye.\n")
# 如果当前模块是主程序，则执行 main() 函数
if __name__ == "__main__":
    main()

######################################################################
#
# Porting Notes
#
#   原始程序在验证玩家输入方面做得非常出色（与本书中许多其他程序相比）。这些检查和响应已经完全复制。
#
#
# Ideas for Modifications
#
#   可能应该在说明中提到最大的猜测次数是 MAX_NUM，不是吗？
#
#   这个程序能否被编写成使用从2到6位数字来猜测？这将如何改变创建“result”字符串的程序？
#
######################################################################
```