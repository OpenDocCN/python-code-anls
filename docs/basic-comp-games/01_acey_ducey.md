# 01 AceyDucey

## 导入

```py
import random
```

## `cards`

```py
# 定义卡牌面值和名称的映射
cards = {
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "Jack",
    11: "Queen",
    12: "King",
    13: "Ace",
}
```

## `get_user_bet()`

```py
# 获取玩家输入的赌金
# 保证它是正数，并且小于等于可用资金
def get_user_bet(cash):
    while True:
        try:
            bet = int(input("What is your bet? "))
            if bet < 0:
                print("Bet must be more than zero")
            elif bet == 0:
                print("CHICKEN!!\n")
            elif bet > cash:
                print("Sorry, my friend but you bet too much")
                print(f"You only have {cash} dollars to bet")
            else:
                return bet
        except ValueError:
            print("Please enter a positive number")
```

## `draw_3cards()`

```py
# 无放回抽三张牌，保证第一张小于第二张
def draw_3cards():
    round_cards = list(cards.keys())
    random.shuffle(round_cards)
    card_a, card_b, card_c = round_cards.pop(), round_cards.pop(), round_cards.pop()
    if card_a > card_b:
        card_a, card_b = card_b, card_a
    return (card_a, card_b, card_c)
```

## `play_game()`

```py
# 游戏的主要逻辑
def play_game():
    """Play the game"""
    cash = 100
    while cash > 0:
        print(f"You now have {cash} dollars\n")
        print("Here are you next two cards")
        # 抽三张牌，展示前两张
        card_a, card_b, card_c = draw_3cards()
        print(f" {cards[card_a]}")
        print(f" {cards[card_b]}\n")
        # 玩家猜测第三张是否在前两张之间，并输入赌金
        bet = get_user_bet(cash)
        # 扣掉赌金，展示第三张
        cash -= bet
        print(f" {cards[card_c]}")
        # 检查猜测结果
        # 如果猜测正确，返还双倍赌金，否则什么也不做
        if card_a < card_c < card_b:
            print("You win!!!")
            cash += bet * 2
        else:
            print("Sorry, you lose")

    # 可用资金为 0，就结束游戏
    print("Sorry, friend, but you blew your wad")
```

## `main()`

```py
# 程序入口
def main():
    # 首先打印游戏介绍
    print("""
Acey-Ducey is played in the following manner
The dealer (computer) deals two cards face up
You have an option to be or not bet depending
on whether or not you feel the card will have
a value between the first two.
If you do not want to bet, input a 0
    """)
    while True:
        # 在循环中开始游戏
        play_game()
        # 游戏结束之后，询问玩家是否继续，不继续就跳出循环
        keep_playing = input("Try again? (yes or no) ").lower() in ["yes", "y"]
        if not keep_playing: break
    print("Ok hope you had fun")


if __name__ == "__main__": main()
```
