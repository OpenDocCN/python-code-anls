# `basic-computer-games\94_War\python\war.py`

```py
#!/usr/bin/env python3
# 指定脚本解释器为 Python 3

"""
WAR

Converted from BASIC to Python by Trevor Hobson
"""
# 多行注释，描述游戏的名称和作者

import json
import random
from pathlib import Path
from typing import List

# 导入所需的模块和类型提示


def card_value(input: str) -> int:
    # 定义函数，根据输入的卡牌返回其值
    return ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"].index(
        input.split("-")[1]
    )
    # 返回卡牌值


def play_game() -> None:
    """Play one round of the game"""
    # 定义函数，玩一轮游戏
    with open(Path(__file__).parent / "cards.json") as f:
        # 打开 cards.json 文件
        cards: List[str] = json.load(f)
        # 读取卡牌列表

    random.shuffle(cards)
    # 随机打乱卡牌顺序
    score_you = 0
    score_computer = 0
    cards_left = 52
    # 初始化玩家和电脑的分数，以及剩余卡牌数量
    for round in range(26):
        # 循环进行 26 轮游戏
        print()
        card_you = cards[round]
        card_computer = cards[round * 2]
        # 获取玩家和电脑的卡牌
        print("You:", card_you, " " * (8 - len(card_you)) + "Computer:", card_computer)
        # 打印玩家和电脑的卡牌
        value_you = card_value(card_you)
        value_computer = card_value(card_computer)
        # 获取玩家和电脑的卡牌值
        if value_you > value_computer:
            score_you += 1
            print(
                "You win. You have", score_you, "and the computer has", score_computer
            )
        elif value_computer > value_you:
            score_computer += 1
            print(
                "The computer wins!!! You have",
                score_you,
                "and the computer has",
                score_computer,
            )
        else:
            print("Tie. No score change.")
        # 判断比较玩家和电脑的卡牌值，更新分数
        cards_left -= 2
        # 更新剩余卡牌数量
        if cards_left > 2 and input("Do you want to continue ").lower().startswith("n"):
            break
        # 判断是否继续游戏
    if cards_left == 0:
        print(
            "\nWe have run out of cards. Final score: You:",
            score_you,
            "the computer:",
            score_computer,
        )
    # 如果卡牌用完，打印最终分数
    print("\nThanks for playing. It was fun.")
    # 打印感谢信息


def main() -> None:
    print(" " * 33 + "WAR")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    print("This is the card game of war. Each card is given by suit-#")
    print("as S-7 for Spade 7.")
    # 打印游戏信息
    # 如果用户输入以字母y开头的字符串，表示用户想要获得游戏指南
    if input("Do you want directions ").lower().startswith("y"):
        # 打印游戏规则说明
        print("The computer gives you and it a 'card'. The higher card")
        print("(numerically) wins. The game ends when you choose not to")
        print("continue or when you have finished the pack.")

    # 设置一个变量来表示是否继续游戏
    keep_playing = True
    # 当变量为True时，循环进行游戏
    while keep_playing:
        # 调用play_game()函数进行游戏
        play_game()
        # 询问用户是否继续游戏，如果输入以字母y开头的字符串，则继续游戏
        keep_playing = input("\nPlay again? (yes or no) ").lower().startswith("y")
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```