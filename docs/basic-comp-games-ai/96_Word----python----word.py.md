# `basic-computer-games\96_Word\python\word.py`

```

#!/usr/bin/env python3
# 指定使用 Python3 解释器

"""
WORD

Converted from BASIC to Python by Trevor Hobson
"""
# 程序的简要说明

import random
# 导入 random 模块

words = [
    "DINKY",
    "SMOKE",
    "WATER",
    "GRASS",
    "TRAIN",
    "MIGHT",
    "FIRST",
    "CANDY",
    "CHAMP",
    "WOULD",
    "CLUMP",
    "DOPEY",
]
# 定义一个包含单词的列表

def play_game() -> None:
    """Play one round of the game"""
    # 定义一个函数，用于进行一轮游戏

    random.shuffle(words)
    # 随机打乱单词列表中的单词顺序
    target_word = words[0]
    # 选择打乱后的列表中的第一个单词作为目标单词
    guess_count = 0
    # 初始化猜测次数为 0
    guess_progress = ["-"] * 5
    # 初始化猜测进度为包含 5 个 "-" 的列表

    print("You are starting a new game...")
    # 打印提示信息
    while True:
        # 进入循环
        guess_word = ""
        # 初始化猜测单词为空字符串
        while guess_word == "":
            # 进入循环，直到猜测单词不为空
            guess_word = input("\nGuess a five letter word. ").upper()
            # 获取用户输入的猜测单词并转换为大写
            if guess_word == "?":
                break
            # 如果用户输入 "?"，则跳出循环
            elif not guess_word.isalpha() or len(guess_word) != 5:
                guess_word = ""
                # 如果用户输入的不是字母或长度不为 5，则重置猜测单词为空字符串，并提示用户重新输入
                print("You must guess a five letter word. Start again.")
        guess_count += 1
        # 猜测次数加一
        if guess_word == "?":
            print("The secret word is", target_word)
            break
            # 如果用户输入 "?"，则打印目标单词并跳出循环
        else:
            common_letters = ""
            matches = 0
            # 初始化公共字母和匹配次数
            for i in range(5):
                for j in range(5):
                    if guess_word[i] == target_word[j]:
                        matches += 1
                        common_letters = common_letters + guess_word[i]
                        if i == j:
                            guess_progress[j] = guess_word[i]
            # 遍历猜测单词和目标单词，找出匹配的字母和位置
            print(
                f"There were {matches}",
                f"matches and the common letters were... {common_letters}",
            )
            # 打印匹配次数和公共字母
            print(
                "From the exact letter matches, you know............ "
                + "".join(guess_progress)
            )
            # 打印猜测进度
            if "".join(guess_progress) == guess_word:
                print(f"\nYou have guessed the word. It took {guess_count} guesses!")
                break
                # 如果猜测进度和猜测单词相同，则打印猜测次数并跳出循环
            elif matches == 0:
                print("\nIf you give up, type '?' for you next guess.")
                # 如果没有匹配，则提示用户可以输入 "?" 放弃猜测


def main() -> None:
    print(" " * 33 + "WORD")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    # 打印游戏标题和信息

    print("I am thinking of a word -- you guess it. I will give you")
    print("clues to help you get it. Good luck!!\n")
    # 打印游戏提示信息

    keep_playing = True
    # 初始化继续游戏标志为 True
    while keep_playing:
        play_game()
        # 调用 play_game 函数进行游戏
        keep_playing = input("\nWant to play again? ").lower().startswith("y")
        # 提示用户是否继续游戏，如果输入以 "y" 开头的字符串，则继续游戏


if __name__ == "__main__":
    main()
    # 如果当前脚本被直接执行，则调用 main 函数开始游戏

```