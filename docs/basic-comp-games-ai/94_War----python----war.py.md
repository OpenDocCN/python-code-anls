# `94_War\python\war.py`

```
#!/usr/bin/env python3  # 指定脚本的解释器为 Python3

"""
WAR

Converted from BASIC to Python by Trevor Hobson
"""

import json  # 导入 json 模块，用于处理 JSON 数据
import random  # 导入 random 模块，用于生成随机数
from pathlib import Path  # 从 pathlib 模块中导入 Path 类，用于处理文件路径
from typing import List  # 从 typing 模块中导入 List 类型，用于声明列表类型


def card_value(input: str) -> int:  # 定义一个函数，参数为字符串类型，返回值为整数类型
    return ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"].index(  # 返回输入字符串在列表中的索引值
        input.split("-")[1]  # 以 "-" 分割输入字符串，并取第二部分
    )
def play_game() -> None:
    """Play one round of the game"""  # 函数注释，说明该函数的作用
    with open(Path(__file__).parent / "cards.json") as f:  # 打开名为 "cards.json" 的文件
        cards: List[str] = json.load(f)  # 从文件中加载数据并存储在名为 cards 的列表中

    random.shuffle(cards)  # 对卡牌列表进行随机洗牌
    score_you = 0  # 初始化玩家得分
    score_computer = 0  # 初始化电脑得分
    cards_left = 52  # 初始化剩余卡牌数量
    for round in range(26):  # 循环进行 26 轮游戏
        print()  # 打印空行
        card_you = cards[round]  # 获取玩家当前轮的卡牌
        card_computer = cards[round * 2]  # 获取电脑当前轮的卡牌
        print("You:", card_you, " " * (8 - len(card_you)) + "Computer:", card_computer)  # 打印玩家和电脑的卡牌
        value_you = card_value(card_you)  # 获取玩家卡牌的点数
        value_computer = card_value(card_computer)  # 获取电脑卡牌的点数
        if value_you > value_computer:  # 判断玩家的点数是否大于电脑的点数
            score_you += 1  # 玩家得分加一
            print(
                "You win. You have", score_you, "and the computer has", score_computer  # 打印玩家得分和电脑得分
        )
```
这是一个if语句的结束标志。

```
        elif value_computer > value_you:
            score_computer += 1
            print(
                "The computer wins!!! You have",
                score_you,
                "and the computer has",
                score_computer,
            )
```
如果电脑的牌面点数大于玩家的牌面点数，则电脑得分加1，并打印出电脑获胜的消息以及当前玩家和电脑的得分。

```
        else:
            print("Tie. No score change.")
```
如果玩家和电脑的牌面点数相同，则打印出平局的消息，不改变得分。

```
        cards_left -= 2
```
每轮游戏结束后，剩余的牌数减去2。

```
        if cards_left > 2 and input("Do you want to continue ").lower().startswith("n"):
            break
```
如果剩余的牌数大于2并且玩家输入的回答以字母'n'开头，则结束游戏。

```
    if cards_left == 0:
        print(
            "\nWe have run out of cards. Final score: You:",
            score_you,
            "the computer:",
            score_computer,
```
如果剩余的牌数为0，则打印出游戏结束的消息以及最终的得分。
        )
    print("\nThanks for playing. It was fun.")


def main() -> None:
    # 打印游戏标题
    print(" " * 33 + "WAR")
    # 打印游戏信息
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    print("This is the card game of war. Each card is given by suit-#")
    print("as S-7 for Spade 7.")
    # 询问是否需要游戏说明
    if input("Do you want directions ").lower().startswith("y"):
        print("The computer gives you and it a 'card'. The higher card")
        print("(numerically) wins. The game ends when you choose not to")
        print("continue or when you have finished the pack.")

    keep_playing = True
    # 循环进行游戏
    while keep_playing:
        play_game()
        # 询问是否继续游戏
        keep_playing = input("\nPlay again? (yes or no) ").lower().startswith("y")
# 如果当前脚本被直接执行，则执行 main() 函数
if __name__ == "__main__":
    main()
```

这段代码用于判断当前脚本是否被直接执行，如果是，则调用 main() 函数。这是一种常见的编程习惯，可以使代码更具可重用性和模块化。
```