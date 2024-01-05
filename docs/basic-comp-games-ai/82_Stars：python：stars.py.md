# `d:/src/tocomm/basic-computer-games\82_Stars\python\stars.py`

```
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
import random  # 导入 random 模块

# Some contants
MAX_NUM = 100  # 定义最大数为 100
MAX_GUESSES = 7  # 定义最大猜测次数为 7


def print_instructions() -> None:
    """Instructions on how to play"""  # 打印游戏说明的函数注释
    print("I am thinking of a whole number from 1 to %d" % MAX_NUM)  # 打印游戏规则
    print("Try to guess my number.  After you guess, I")  # 打印游戏规则
    print("will type one or more stars (*).  The more")  # 打印游戏规则
    print("stars I type, the closer you are to my number.")  # 打印游戏规则
    print("one star (*) means far away, seven stars (*******)")  # 打印游戏规则
    print("means really close!  You get %d guesses." % MAX_GUESSES)  # 打印游戏规则
def print_stars(secret_number, guess) -> None:
    # 计算猜测值与秘密数字的差值
    diff = abs(guess - secret_number)
    stars = ""
    # 根据差值生成星号字符串
    for i in range(8):
        if diff < 2**i:
            stars += "*"
    # 打印星号字符串
    print(stars)


def get_guess(prompt: str) -> int:
    # 循环直到用户输入有效的猜测值
    while True:
        guess_str = input(prompt)
        # 检查用户输入是否为数字
        if guess_str.isdigit():
            guess = int(guess_str)
            return guess


def main() -> None:
    # 显示介绍文本
    print("\n                   Stars")
    print("Creative Computing  Morristown, New Jersey")  # 打印 Creative Computing  Morristown, New Jersey
    print("\n\n")  # 打印两个空行
    # "*** Stars - People's Computer Center, MenloPark, CA"  # 注释，说明这是一条注释

    response = input("Do you want instructions? ")  # 获取用户输入，询问是否需要说明
    if response.upper()[0] == "Y":  # 将用户输入转换为大写并取第一个字符，判断是否为Y
        print_instructions()  # 如果用户需要说明，则打印说明

    still_playing = True  # 设置游戏状态为正在进行
    while still_playing:  # 进入游戏循环

        # "*** Computer thinks of a number"  # 注释，说明计算机正在想一个数字
        secret_number = random.randint(1, MAX_NUM)  # 生成一个1到MAX_NUM之间的随机数作为秘密数字
        print("\n\nOK, I am thinking of a number, start guessing.")  # 打印提示信息，开始猜数字

        # Init/start guess loop  # 初始化/开始猜数字循环
        guess_number = 0  # 初始化猜测次数
        player_has_won = False  # 初始化玩家是否获胜的状态为False
        while (guess_number < MAX_GUESSES) and not player_has_won:  # 进入猜数字循环，条件是猜测次数小于最大猜测次数且玩家未获胜
            # 打印空行
            print()
            # 获取玩家猜测的数字
            guess = get_guess("Your guess? ")
            # 猜测次数加一
            guess_number += 1

            # 如果猜对了
            if guess == secret_number:
                # 标记玩家已经赢了
                player_has_won = True
                # 打印祝贺信息
                print("**************************************************!!!")
                print(f"You got it in {guess_number} guesses!!!")

            # 如果猜错了
            else:
                # 打印星号提示
                print_stars(secret_number, guess)

            # 猜测循环结束

        # 如果玩家没有赢
        if not player_has_won:
            # 打印未猜中的提示信息
            print(f"\nSorry, that's {guess_number} guesses, number was {secret_number}")

        # 继续游戏？
        response = input("\nPlay again? ")  # 询问用户是否要再玩一次游戏，并将用户输入的内容赋值给变量response
        if response.upper()[0] != "Y":  # 如果用户输入的内容的大写形式的第一个字符不是Y
            still_playing = False  # 将变量still_playing设置为False，结束游戏


if __name__ == "__main__":  # 如果当前文件被直接运行
    main()  # 调用main函数

######################################################################
#
# Porting Notes
#
#   The original program never exited--it just kept playing rounds
#   over and over.  This version asks to continue each time.
#   原始程序从未退出--它只是一直重复玩游戏。这个版本每次都会询问是否继续。
#
#
# Ideas for Modifications
#
#   Let the player know how many guesses they have remaining after
#   each incorrect guess.
#   让玩家在每次猜错后知道他们还剩下多少次猜测的机会。
# 询问玩家在游戏开始时选择技能级别，这将影响 MAX_NUM 和 MAX_GUESSES 的值。
# 例如：
#
#   简单 = 8 次猜测，1 到 50
#   中等 = 7 次猜测，1 到 100
#   困难 = 6 次猜测，1 到 200
#
# ######################################################################
```