# `basic-computer-games\01_Acey_Ducey\python\acey_ducey_oo.py`

```

"""
AceyDuchy
From: BASIC Computer Games (1978)
      Edited by David Ahl
"The original BASIC program author was Bill Palmby
 of Prairie View, Illinois."
Python port by Aviyam Fischer, 2022
"""

# 导入必要的类型提示模块
from typing import List, Literal, NamedTuple, TypeAlias, get_args
# 导入随机数模块
import random

# 定义类型别名
Suit: TypeAlias = Literal["\u2665", "\u2666", "\u2663", "\u2660"]
Rank: TypeAlias = Literal[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# 定义卡片类
class Card(NamedTuple):
    suit: Suit
    rank: Rank

    # 重写打印方法
    def __str__(self) -> str:
        r = str(self.rank)
        r = {"11": "J", "12": "Q", "13": "K", "14": "A"}.get(r, r)
        return f"{r}{self.suit}"

# 定义牌组类
class Deck:
    def __init__(self) -> None:
        self.cards: List[Card] = []
        self.build()

    # 创建一副牌
    def build(self) -> None:
        for suit in get_args(Suit):
            for rank in get_args(Rank):
                self.cards.append(Card(suit, rank))

    # 洗牌
    def shuffle(self) -> None:
        random.shuffle(self.cards)

    # 发牌
    def deal(self) -> Card:
        return self.cards.pop()

# 定义游戏类
class Game:
    def __init__(self) -> None:
        self.deck = Deck()
        self.deck.shuffle()
        self.card_a = self.deck.deal()
        self.card_b = self.deck.deal()
        self.money = 100
        self.not_done = True

    # 游戏进行
    def play(self) -> None:
        while self.not_done:
            while self.money > 0:
                card_a = self.card_a
                card_b = self.card_b

                if card_a.rank > card_b.rank:
                    card_a, card_b = card_b, card_a

                if card_a.rank == card_b.rank:
                    self.card_b = self.deck.deal()
                    card_b = self.card_b

                print(f"You have:\t ${self.money} ")
                print(f"Your cards:\t {card_a} {card_b}")

                bet = int(input("What is your bet? "))
                player_card = self.deck.deal()
                if 0 < bet <= self.money:

                    print(f"Your deal:\t {player_card}")
                    if card_a.rank <= player_card.rank <= card_b.rank:
                        print("You Win!")
                        self.money += bet
                    else:
                        print("You Lose!")
                        self.money -= bet
                        self.not_done = False
                else:
                    print("Chicken!")
                    print(f"Your deal should have been: {player_card}")
                    if card_a.rank < player_card.rank < card_b.rank:
                        print("You could have won!")
                    else:
                        print("You would lose, so it was wise of you to chicken out!")

                if len(self.deck.cards) <= 3:
                    print("You ran out of cards. Game over.")
                    self.not_done = False
                    break

                self.card_a = self.deck.deal()
                self.card_b = self.deck.deal()

        if self.money == 0:
            self.not_done = False

# 游戏循环
def game_loop() -> None:
    game_over = False

    while not game_over:
        game = Game()
        game.play()
        print(f"You have ${game.money} left")
        print("Would you like to play again? (y/n)")
        if input() == "n":
            game_over = True

# 主函数
def main() -> None:
    print(
        """
    Acey Ducey is a card game where you play against the computer.
    The Dealer(computer) will deal two cards facing up.
    You have an option to bet or not bet depending on whether or not you
    feel the card will have a value between the first two.
    If you do not want to bet input a 0
    """
    )
    game_loop()
    print("\nThanks for playing!")

# 程序入口
if __name__ == "__main__":
    random.seed()
    main()

```