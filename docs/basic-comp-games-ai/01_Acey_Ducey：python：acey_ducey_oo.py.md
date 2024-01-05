# `d:/src/tocomm/basic-computer-games\01_Acey_Ducey\python\acey_ducey_oo.py`

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

# 定义卡片类，包含花色和点数
class Card(NamedTuple):
    suit: Suit
    rank: Rank
    def __str__(self) -> str:  # 定义一个方法，返回类型为字符串
        r = str(self.rank)  # 将牌的点数转换为字符串
        r = {"11": "J", "12": "Q", "13": "K", "14": "A"}.get(r, r)  # 如果点数是特殊牌（11、12、13、14），则替换为对应的字母
        return f"{r}{self.suit}"  # 返回点数和花色组成的字符串


class Deck:  # 定义一个类 Deck
    def __init__(self) -> None:  # 初始化方法
        self.cards: List[Card] = []  # 创建一个空列表，用于存放卡牌
        self.build()  # 调用 build 方法，构建一副完整的扑克牌

    def build(self) -> None:  # 定义一个方法，用于构建一副完整的扑克牌
        for suit in get_args(Suit):  # 遍历花色
            for rank in get_args(Rank):  # 遍历点数
                self.cards.append(Card(suit, rank))  # 创建一张新的卡牌，并加入到卡牌列表中

    def shuffle(self) -> None:  # 定义一个方法，用于洗牌
        random.shuffle(self.cards)  # 使用 random 模块的 shuffle 方法对卡牌列表进行洗牌

    def deal(self) -> Card:  # 定义一个方法，用于发牌
        return self.cards.pop()
# 从牌堆中弹出一张牌并返回

class Game:
    def __init__(self) -> None:
        self.deck = Deck()
        self.deck.shuffle()
        self.card_a = self.deck.deal()
        self.card_b = self.deck.deal()
        self.money = 100
        self.not_done = True
# 初始化游戏对象，创建一副牌并洗牌，发给玩家两张牌，设置初始资金为100，游戏状态为未结束

    def play(self) -> None:
        while self.not_done:
            while self.money > 0:
                card_a = self.card_a
                card_b = self.card_b
# 当游戏未结束且资金大于0时，进行游戏循环，获取玩家手中的两张牌

                if card_a.rank > card_b.rank:
                    card_a, card_b = card_b, card_a
# 如果第一张牌的点数大于第二张牌的点数，则交换两张牌的位置
                # 如果两张牌的点数相同，则重新发一张牌给玩家B
                if card_a.rank == card_b.rank:
                    self.card_b = self.deck.deal()
                    card_b = self.card_b

                # 打印玩家的余额
                print(f"You have:\t ${self.money} ")
                # 打印玩家手中的牌
                print(f"Your cards:\t {card_a} {card_b}")

                # 玩家下注
                bet = int(input("What is your bet? "))
                player_card = self.deck.deal()
                # 如果下注金额在玩家的余额范围内
                if 0 < bet <= self.money:

                    # 打印玩家的发牌
                    print(f"Your deal:\t {player_card}")
                    # 如果玩家的发牌在两张庄家的牌之间
                    if card_a.rank <= player_card.rank <= card_b.rank:
                        print("You Win!")
                        # 玩家赢得下注金额
                        self.money += bet
                    else:
                        print("You Lose!")
                        # 玩家失去下注金额
                        self.money -= bet
                        # 游戏结束
                        self.not_done = False
                else:  # 如果不满足上述条件
                    print("Chicken!")  # 打印“Chicken!”
                    print(f"Your deal should have been: {player_card}")  # 打印玩家应该出的牌
                    if card_a.rank < player_card.rank < card_b.rank:  # 如果玩家出的牌在两张已经出的牌之间
                        print("You could have won!")  # 打印“你本可以赢！”
                    else:  # 如果玩家出的牌不在两张已经出的牌之间
                        print("You would lose, so it was wise of you to chicken out!")  # 打印“你会输，所以你明智地选择了退出！”
                
                if len(self.deck.cards) <= 3:  # 如果剩余牌数小于等于3
                    print("You ran out of cards. Game over.")  # 打印“你的牌已经用完。游戏结束。”
                    self.not_done = False  # 将游戏状态标记为结束
                    break  # 结束循环

                self.card_a = self.deck.deal()  # 从牌堆中发一张牌给card_a
                self.card_b = self.deck.deal()  # 从牌堆中发一张牌给card_b

        if self.money == 0:  # 如果玩家的钱为0
            self.not_done = False  # 将游戏状态标记为结束
def game_loop() -> None:  # 定义一个名为game_loop的函数，不返回任何值
    game_over = False  # 创建一个名为game_over的变量，并将其值设置为False

    while not game_over:  # 当game_over的值为False时执行以下代码
        game = Game()  # 创建一个名为game的对象，调用Game类的构造函数
        game.play()  # 调用game对象的play方法
        print(f"You have ${game.money} left")  # 打印字符串，其中包含game对象的money属性的值
        print("Would you like to play again? (y/n)")  # 打印提示信息
        if input() == "n":  # 获取用户输入，如果输入为"n"，执行以下代码
            game_over = True  # 将game_over的值设置为True


def main() -> None:  # 定义一个名为main的函数，不返回任何值
    print(  # 打印多行字符串
        """
    Acey Ducey is a card game where you play against the computer.
    The Dealer(computer) will deal two cards facing up.
    You have an option to bet or not bet depending on whether or not you
    feel the card will have a value between the first two.
    If you do not want to bet input a 0
```  # 多行字符串的内容
    """
    )  # 该行代码是一个字符串结尾的注释，没有实际作用，可能是代码中的一个错误
    game_loop()  # 调用名为game_loop的函数，开始游戏循环
    print("\nThanks for playing!")  # 打印感谢信息
```


```python
if __name__ == "__main__":
    random.seed()  # 使用系统时间作为随机数种子，确保每次运行的随机数不同
    main()  # 调用名为main的函数，开始程序的主要逻辑
```