# `basic-computer-games\94_War\python\war.py`

```

#!/usr/bin/env python3
# 指定使用 Python3 解释器

"""
WAR

Converted from BASIC to Python by Trevor Hobson
"""
# 程序的简要介绍

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

def play_game() -> None:
    """Play one round of the game"""
    # 定义函数，玩一轮游戏
    with open(Path(__file__).parent / "cards.json") as f:
        # 打开 cards.json 文件
        cards: List[str] = json.load(f)
        # 读取卡牌数据

    random.shuffle(cards)
    # 洗牌
    score_you = 0
    score_computer = 0
    cards_left = 52
    # 初始化得分和剩余卡牌数量
    for round in range(26):
        # 进行26轮游戏
        print()
        card_you = cards[round]
        card_computer = cards[round * 2]
        # 获取玩家和电脑的卡牌
        print("You:", card_you, " " * (8 - len(card_you)) + "Computer:", card_computer)
        value_you = card_value(card_you)
        value_computer = card_value(card_computer)
        # 获取卡牌的值
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
        cards_left -= 2
        # 更新得分和剩余卡牌数量
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
    print("\nThanks for playing. It was fun.")
    # 游戏结束后输出结果

def main() -> None:
    print(" " * 33 + "WAR")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    print("This is the card game of war. Each card is given by suit-#")
    print("as S-7 for Spade 7.")
    # 输出游戏介绍
    if input("Do you want directions ").lower().startswith("y"):
        print("The computer gives you and it a 'card'. The higher card")
        print("(numerically) wins. The game ends when you choose not to")
        print("continue or when you have finished the pack.")
        # 输出游戏规则

    keep_playing = True
    while keep_playing:
        play_game()
        keep_playing = input("\nPlay again? (yes or no) ").lower().startswith("y")
        # 循环进行游戏

if __name__ == "__main__":
    main()
    # 调用主函数

```