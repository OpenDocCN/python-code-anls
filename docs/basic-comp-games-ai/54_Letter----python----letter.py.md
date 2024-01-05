# `54_Letter\python\letter.py`

```
"""
LETTER

A letter guessing game.

Ported by Dave LeCompte
"""

import random

# The original code printed character 7, the "BELL" character 15 times
# when the player won. Many modern systems do not support this, and in
# any case, it can quickly become annoying, so it is disabled here.

BELLS_ON_SUCCESS = False  # 设置变量BELLS_ON_SUCCESS为False，表示游戏胜利时不会发出“BELL”声音


def print_instructions() -> None:
    print("LETTER GUESSING GAME")  # 打印游戏标题
    print()  # 打印空行
    # 打印游戏提示信息
    print("I'LL THINK OF A LETTER OF THE ALPHABET, A TO Z.")
    print("TRY TO GUESS MY LETTER AND I'LL GIVE YOU CLUES")
    print("AS TO HOW CLOSE YOU'RE GETTING TO MY LETTER.")


def play_game() -> None:
    # 生成一个随机字母的 ASCII 值作为目标值
    target_value = random.randint(ord("A"), ord("Z"))
    num_guesses = 0
    # 打印游戏开始提示信息
    print()
    print("O.K., I HAVE A LETTER.  START GUESSING.")
    print()
    # 循环进行猜字母的游戏
    while True:
        # 提示用户输入猜测的字母
        print("WHAT IS YOUR GUESS?")
        num_guesses += 1
        # 获取用户输入字母的 ASCII 值
        guess = ord(input())
        print()
        # 判断用户猜测的字母是否与目标字母相同
        if guess == target_value:
            # 打印猜中目标字母的提示信息
            print()
            print(f"YOU GOT IT IN {num_guesses} GUESSES!!")
            # 如果猜测次数超过5次，给予额外提示
            if num_guesses > 5:
                print("BUT IT SHOULDN'T TAKE MORE THAN 5 GUESSES!")
                # 打印提示信息，提醒玩家不应该需要超过5次猜测
                # 跳转到515行
            print("GOOD JOB !!!!!")
            # 打印祝贺信息

            if BELLS_ON_SUCCESS:
                bell_str = chr(7) * 15
                # 如果成功时需要响铃，则创建一个包含15个响铃字符的字符串
                print(bell_str)

            print()
            print("LET'S PLAY AGAIN.....")
            # 打印提示信息，表示再次开始游戏
            return
        elif guess > target_value:
            print("TOO HIGH. TRY A LOWER LETTER.")
            # 如果猜测值大于目标值，则打印提示信息，要求猜测更小的值
            continue
        else:
            print("TOO LOW. TRY A HIGHER LETTER.")
            # 如果猜测值小于目标值，则打印提示信息，要求猜测更大的值
            continue


def main() -> None:
    print(" " * 33 + "LETTER")  # 在屏幕上打印出 "LETTER"，并在前面添加 33 个空格
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")  # 在屏幕上打印出 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并在前面添加 15 个空格，然后换行两次

    print_instructions()  # 调用函数 print_instructions()，打印游戏的说明

    while True:  # 进入一个无限循环
        play_game()  # 调用函数 play_game()，开始游戏

if __name__ == "__main__":  # 如果当前脚本被直接执行，则执行下面的代码
    main()  # 调用函数 main()，开始执行主程序
```