# `basic-computer-games\54_Letter\python\letter.py`

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

BELLS_ON_SUCCESS = False  # 设置是否在玩家猜对时响铃的标志


def print_instructions() -> None:
    # 打印游戏说明
    print("LETTER GUESSING GAME")
    print()
    print("I'LL THINK OF A LETTER OF THE ALPHABET, A TO Z.")
    print("TRY TO GUESS MY LETTER AND I'LL GIVE YOU CLUES")
    print("AS TO HOW CLOSE YOU'RE GETTING TO MY LETTER.")


def play_game() -> None:
    # 生成目标字母的 ASCII 值
    target_value = random.randint(ord("A"), ord("Z"))
    num_guesses = 0  # 猜测次数初始化为 0
    print()
    print("O.K., I HAVE A LETTER.  START GUESSING.")
    print()
    while True:
        print("WHAT IS YOUR GUESS?")
        num_guesses += 1  # 猜测次数加一
        guess = ord(input())  # 获取用户输入字母的 ASCII 值
        print()
        if guess == target_value:  # 如果猜对了
            print()
            print(f"YOU GOT IT IN {num_guesses} GUESSES!!")  # 打印猜对的次数
            if num_guesses > 5:  # 如果猜对次数大于 5
                print("BUT IT SHOULDN'T TAKE MORE THAN 5 GUESSES!")  # 提示不应该超过 5 次
            print("GOOD JOB !!!!!")  # 打印祝贺语句

            if BELLS_ON_SUCCESS:  # 如果设置了响铃标志
                bell_str = chr(7) * 15  # 生成响铃字符串
                print(bell_str)  # 打印响铃字符串

            print()
            print("LET'S PLAY AGAIN.....")  # 打印再玩一次的提示
            return  # 结束当前游戏
        elif guess > target_value:  # 如果猜的值比目标值大
            print("TOO HIGH. TRY A LOWER LETTER.")  # 提示猜的值太大
            continue  # 继续循环
        else:  # 如果猜的值比目标值小
            print("TOO LOW. TRY A HIGHER LETTER.")  # 提示猜的值太小
            continue  # 继续循环


def main() -> None:
    print(" " * 33 + "LETTER")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

    print_instructions()  # 打印游戏说明

    while True:
        play_game()  # 开始游戏


if __name__ == "__main__":
    main()  # 执行主函数

```