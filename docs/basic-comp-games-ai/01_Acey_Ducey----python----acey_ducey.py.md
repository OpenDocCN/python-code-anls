# `basic-computer-games\01_Acey_Ducey\python\acey_ducey.py`

```
#!/usr/bin/env python3
"""
Play the Acey-Ducey game
https://www.atariarchives.org/basicgames/showpage.php?page=2
"""

import random


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


def play_game() -> None:
    # 初始化玩家的现金
    cash = 100
    # 当玩家还有现金时，循环进行游戏
    while cash > 0:
        print(f"You now have {cash} dollars\n")
        print("Here are you next two cards")
        round_cards = list(cards.keys())  # 从字典中获取卡片
        card_a = random.choice(round_cards)  # 随机选择一张卡片
        card_b = card_a  # 克隆第一张卡片，以避免第二张卡片与第一张相同
        while (card_a == card_b):  # 如果两张卡片相同，则选择另一张卡片
            card_b = random.choice(round_cards)
        card_c = random.choice(round_cards)  # 选择最后一张卡片
        if card_a > card_b:  # 如果第一张卡片大于第二张卡片，则交换它们的位置
            card_a, card_b = card_b, card_a
        print(f" {cards[card_a]}")
        print(f" {cards[card_b]}\n")
        while True:
            try:
                bet = int(input("What is your bet? "))  # 获取玩家的赌注
                if bet < 0:
                    raise ValueError("Bet must be more than zero")  # 如果赌注小于零，则引发异常
                if bet == 0:
                    print("CHICKEN!!\n")  # 如果赌注为零，则输出提示信息
                if bet > cash:
                    print("Sorry, my friend but you bet too much")  # 如果赌注大于现金数，则输出提示信息
                    print(f"You only have {cash} dollars to bet")
                    continue
                cash -= bet  # 扣除赌注金额
                break

            except ValueError:
                print("Please enter a positive number")  # 如果输入不是正数，则输出提示信息
        print(f" {cards[card_c]}")
        if bet > 0:
            if card_a <= card_c <= card_b:  # 如果第一张卡片小于等于第三张卡片小于等于第二张卡片
                print("You win!!!")  # 输出赢得游戏的提示信息
                cash += bet * 2  # 增加现金数
            else:
                print("Sorry, you lose")  # 输出输掉游戏的提示信息

    print("Sorry, friend, but you blew your wad")  # 输出玩家输光所有现金的提示信息


def main() -> None:
    # 打印三个双引号，用于多行注释的起始
# Acey-Ducey 游戏规则说明
# 庄家（计算机）发两张牌，正面朝上
# 你可以选择下注或不下注，取决于你是否认为第三张牌的值会介于前两张之间
# 如果你不想下注，输入 0
# 游戏保持进行的标志
keep_playing = True

# 当游戏保持进行时，循环执行游戏
while keep_playing:
    play_game()  # 执行游戏
    keep_playing = input("Try again? (yes or no) ").lower().startswith("y")  # 询问是否继续游戏，如果以 "y" 开头则继续

# 打印结束语
print("Ok hope you had fun")

# 如果作为主程序运行，则初始化随机数种子并执行主函数
if __name__ == "__main__":
    random.seed()  # 初始化随机数种子
    main()  # 执行主函数
```