# `basic-computer-games\01_Acey_Ducey\python\acey_ducey.py`

```

#!/usr/bin/env python3
"""
Play the Acey-Ducey game
https://www.atariarchives.org/basicgames/showpage.php?page=2
"""

import random


# 定义扑克牌的字典，键为牌面数字，值为对应的牌面名称
cards = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "Jack",
    12: "Queen",
    13: "King",
    14: "Ace",
}


# 定义游戏函数
def play_game() -> None:
    cash = 100  # 初始金额为100
    while cash > 0:  # 当金额大于0时循环进行游戏
        print(f"You now have {cash} dollars\n")  # 打印当前金额
        print("Here are you next two cards")  # 提示下两张牌
        round_cards = list(cards.keys())  # 从字典中获取所有牌面数字
        card_a = random.choice(round_cards)  # 随机选择一张牌
        card_b = card_a  # 复制第一张牌，避免第二张牌与第一张相同
        while (card_a == card_b):  # 如果两张牌相同，则重新选择第二张牌
            card_b = random.choice(round_cards)
        card_c = random.choice(round_cards)  # 随机选择第三张牌
        if card_a > card_b:  # 如果第一张牌大于第二张牌，则交换它们的位置
            card_a, card_b = card_b, card_a
        print(f" {cards[card_a]}")  # 打印第一张牌的名称
        print(f" {cards[card_b]}\n")  # 打印第二张牌的名称
        while True:
            try:
                bet = int(input("What is your bet? "))  # 输入赌注
                if bet < 0:  # 如果赌注小于0，则抛出异常
                    raise ValueError("Bet must be more than zero")
                if bet == 0:  # 如果赌注为0，则提示“CHICKEN!!”
                    print("CHICKEN!!\n")
                if bet > cash:  # 如果赌注大于当前金额，则提示赌注过大
                    print("Sorry, my friend but you bet too much")
                    print(f"You only have {cash} dollars to bet")
                    continue
                cash -= bet  # 扣除赌注金额
                break

            except ValueError:
                print("Please enter a positive number")  # 输入非正数时提示重新输入
        print(f" {cards[card_c]}")  # 打印第三张牌的名称
        if bet > 0:  # 如果赌注大于0
            if card_a <= card_c <= card_b:  # 如果第三张牌的数字在第一张和第二张牌之间
                print("You win!!!")  # 赢得游戏
                cash += bet * 2  # 赢得的赌注翻倍
            else:
                print("Sorry, you lose")  # 输掉游戏

    print("Sorry, friend, but you blew your wad")  # 当金额为0时，游戏结束


# 主函数
def main() -> None:
    print(
        """
Acey-Ducey is played in the following manner
The dealer (computer) deals two cards face up
You have an option to bet or not bet depending
on whether or not you feel the card will have
a value between the first two.
If you do not want to bet, input a 0
  """
    )  # 打印游戏规则
    keep_playing = True  # 设置继续游戏的标志为True

    while keep_playing:  # 当继续游戏的标志为True时循环进行游戏
        play_game()  # 进行游戏
        keep_playing = input("Try again? (yes or no) ").lower().startswith("y")  # 询问是否继续游戏
    print("Ok hope you had fun")  # 结束游戏后打印提示信息


if __name__ == "__main__":
    random.seed()  # 初始化随机数种子
    main()  # 调用主函数进行游戏

```