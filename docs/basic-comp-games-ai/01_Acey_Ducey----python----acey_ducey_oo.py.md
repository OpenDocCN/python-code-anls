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

# 定义花色类型别名
Suit: TypeAlias = Literal["\u2665", "\u2666", "\u2663", "\u2660"]
# 定义牌面大小类型别名
Rank: TypeAlias = Literal[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# 定义卡片类
class Card(NamedTuple):
    suit: Suit
    rank: Rank

    # 重写卡片类的字符串表示方法
    def __str__(self) -> str:
        # 将牌面大小转换为对应的字符表示
        r = str(self.rank)
        r = {"11": "J", "12": "Q", "13": "K", "14": "A"}.get(r, r)
        return f"{r}{self.suit}"

# 定义牌组类
class Deck:
    def __init__(self) -> None:
        # 初始化牌组为空列表
        self.cards: List[Card] = []
        # 构建牌组
        self.build()

    # 构建牌组的方法
    def build(self) -> None:
        # 遍历所有花色和牌面大小，生成一副完整的牌组
        for suit in get_args(Suit):
            for rank in get_args(Rank):
                self.cards.append(Card(suit, rank))

    # 洗牌的方法
    def shuffle(self) -> None:
        # 使用随机数模块中的shuffle方法洗牌
        random.shuffle(self.cards)

    # 发牌的方法
    def deal(self) -> Card:
        # 从牌组中抽取一张牌并返回
        return self.cards.pop()

# 定义游戏类
class Game:
    def __init__(self) -> None:
        # 初始化游戏时创建一副牌并洗牌
        self.deck = Deck()
        self.deck.shuffle()
        # 发两张牌给玩家
        self.card_a = self.deck.deal()
        self.card_b = self.deck.deal()
        # 初始化玩家的资金和游戏状态
        self.money = 100
        self.not_done = True
    # 定义一个方法，用于进行游戏
    def play(self) -> None:
        # 当游戏未结束时进行循环
        while self.not_done:
            # 当玩家还有钱时进行循环
            while self.money > 0:
                # 获取玩家手中的两张牌
                card_a = self.card_a
                card_b = self.card_b

                # 如果第一张牌的等级大于第二张牌的等级，则交换两张牌
                if card_a.rank > card_b.rank:
                    card_a, card_b = card_b, card_a

                # 如果两张牌的等级相同，则重新发一张牌给玩家
                if card_a.rank == card_b.rank:
                    self.card_b = self.deck.deal()
                    card_b = self.card_b

                # 打印玩家当前的资金和手中的两张牌
                print(f"You have:\t ${self.money} ")
                print(f"Your cards:\t {card_a} {card_b}")

                # 玩家下注
                bet = int(input("What is your bet? "))
                player_card = self.deck.deal()
                # 如果下注金额在合理范围内
                if 0 < bet <= self.money:

                    # 打印玩家的牌
                    print(f"Your deal:\t {player_card}")
                    # 如果玩家的牌在两张牌之间，则玩家赢得下注金额
                    if card_a.rank <= player_card.rank <= card_b.rank:
                        print("You Win!")
                        self.money += bet
                    else:
                        print("You Lose!")
                        self.money -= bet
                        self.not_done = False
                else:
                    # 如果下注金额不在合理范围内
                    print("Chicken!")
                    print(f"Your deal should have been: {player_card}")
                    # 如果玩家的牌在两张牌之间，则玩家本来可以赢得比赛
                    if card_a.rank < player_card.rank < card_b.rank:
                        print("You could have won!")
                    else:
                        print("You would lose, so it was wise of you to chicken out!")

                # 如果牌堆中的牌数量小于等于3，则游戏结束
                if len(self.deck.cards) <= 3:
                    print("You ran out of cards. Game over.")
                    self.not_done = False
                    break

                # 重新发牌给玩家
                self.card_a = self.deck.deal()
                self.card_b = self.deck.deal()

        # 如果玩家的资金为0，则游戏结束
        if self.money == 0:
            self.not_done = False
# 定义游戏循环函数，没有返回值
def game_loop() -> None:
    # 初始化游戏结束标志为 False
    game_over = False

    # 当游戏未结束时循环执行以下代码
    while not game_over:
        # 创建游戏对象
        game = Game()
        # 开始游戏
        game.play()
        # 打印玩家剩余的金额
        print(f"You have ${game.money} left")
        # 打印询问是否再玩一次
        print("Would you like to play again? (y/n)")
        # 如果输入为 "n"，则将游戏结束标志设为 True
        if input() == "n":
            game_over = True


# 定义主函数，没有返回值
def main() -> None:
    # 打印游戏规则说明
    print(
        """
    Acey Ducey is a card game where you play against the computer.
    The Dealer(computer) will deal two cards facing up.
    You have an option to bet or not bet depending on whether or not you
    feel the card will have a value between the first two.
    If you do not want to bet input a 0
    """
    )
    # 调用游戏循环函数
    game_loop()
    # 打印感谢信息
    print("\nThanks for playing!")


# 如果当前脚本为主程序时，执行以下代码
if __name__ == "__main__":
    # 初始化随机数生成器
    random.seed()
    # 调用主函数
    main()
```