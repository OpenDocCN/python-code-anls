# `basic-computer-games\82_Stars\python\stars.py`

```py
"""
Stars

From: BASIC Computer Games (1978)
      Edited by David H. Ahl

"In this game, the computer selects a random number from 1 to 100
 (or any value you set [for MAX_NUM]).  You try to guess the number
 and the computer gives you clues to tell you how close you're
 getting.  One star (*) means you're far away from the number; seven
 stars (*******) means you're really close.  You get 7  guesses.

"On the surface this game is very similar to GUESS; however, the
 guessing strategy is quite different.  See if you can come up with
 one or more approaches to finding the mystery number.

"Bob Albrecht of People's Computer Company created this game."


Python port by Jeff Jetton, 2019
"""


import random

# Some contants
MAX_NUM = 100  # 定义最大数值为100
MAX_GUESSES = 7  # 定义最大猜测次数为7


def print_instructions() -> None:
    """Instructions on how to play"""
    # 打印游戏玩法说明
    print("I am thinking of a whole number from 1 to %d" % MAX_NUM)
    print("Try to guess my number.  After you guess, I")
    print("will type one or more stars (*).  The more")
    print("stars I type, the closer you are to my number.")
    print("one star (*) means far away, seven stars (*******)")
    print("means really close!  You get %d guesses." % MAX_GUESSES)


def print_stars(secret_number, guess) -> None:
    # 打印星星，表示猜测的接近程度
    diff = abs(guess - secret_number)
    stars = ""
    for i in range(8):
        if diff < 2**i:
            stars += "*"
    print(stars)


def get_guess(prompt: str) -> int:
    # 获取用户输入的猜测
    while True:
        guess_str = input(prompt)
        if guess_str.isdigit():
            guess = int(guess_str)
            return guess


def main() -> None:
    # Display intro text
    # 显示游戏介绍文本
    print("\n                   Stars")
    print("Creative Computing  Morristown, New Jersey")
    print("\n\n")
    # "*** Stars - People's Computer Center, MenloPark, CA"

    response = input("Do you want instructions? ")
    if response.upper()[0] == "Y":
        print_instructions()

    still_playing = True
    # 当仍在玩游戏时执行以下代码块
    while still_playing:

        # 电脑随机生成一个数字
        secret_number = random.randint(1, MAX_NUM)
        print("\n\nOK, I am thinking of a number, start guessing.")

        # 初始化/开始猜测循环
        guess_number = 0
        player_has_won = False
        while (guess_number < MAX_GUESSES) and not player_has_won:

            print()
            # 获取玩家的猜测
            guess = get_guess("Your guess? ")
            guess_number += 1

            if guess == secret_number:
                # 玩家猜对了
                player_has_won = True
                print("**************************************************!!!")
                print(f"You got it in {guess_number} guesses!!!")

            else:
                # 打印星号提示玩家
                print_stars(secret_number, guess)

            # 猜测循环结束

        # 玩家未在最大猜测次数内猜对
        if not player_has_won:
            print(f"\nSorry, that's {guess_number} guesses, number was {secret_number}")

        # 继续玩游戏？
        response = input("\nPlay again? ")
        if response.upper()[0] != "Y":
            still_playing = False
# 如果当前模块是主程序，则执行 main() 函数
if __name__ == "__main__":
    main()

######################################################################
#
# 程序移植注意事项
#
#   原始程序从不退出--它只是一直循环播放回合。这个版本在每次询问后都会要求继续。
#
#
# 修改的想法
#
#   每次猜错后让玩家知道他们还有多少次猜测机会。
#
#   在游戏开始时要求玩家选择一个技能级别，这将影响 MAX_NUM 和 MAX_GUESSES 的值。
#   例如：
#
#       简单   = 8 次猜测，1 到 50
#       中等 = 7 次猜测，1 到 100
#       困难   = 6 次猜测，1 到 200
#
######################################################################
```