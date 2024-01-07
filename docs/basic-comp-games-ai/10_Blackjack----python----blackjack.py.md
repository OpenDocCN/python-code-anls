# `basic-computer-games\10_Blackjack\python\blackjack.py`

```

"""
Blackjack

Ported by Martin Thoma in 2022,
using the rust implementation of AnthonyMichaelTDM
"""

# 导入必要的库
import enum
import random
from dataclasses import dataclass
from typing import List, NamedTuple

# 定义玩家类型
class PlayerType(enum.Enum):
    Player = "Player"
    Dealer = "Dealer"

# 定义玩法
class Play(enum.Enum):
    Stand = enum.auto()
    Hit = enum.auto()
    DoubleDown = enum.auto()
    Split = enum.auto()

# 定义卡牌
class Card(NamedTuple):
    name: str

    @property
    def value(self) -> int:
        """
        返回与传入名称的卡片相关联的值
        如果传入的卡片名称不存在，则返回0
        """
        return {
            "ACE": 11,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "10": 10,
            "JACK": 10,
            "QUEEN": 10,
            "KING": 10,
        }.get(self.name, 0)

# 定义手牌
class Hand(NamedTuple):
    cards: List[Card]

    def add_card(self, card: Card) -> None:
        """将传入的卡片添加到这个手牌中"""
        self.cards.append(card)

    def get_total(self) -> int:
        """返回这个手牌中卡片的总点数"""
        total: int = 0
        for card in self.cards:
            total += int(card.value)

        # 如果有一张ACE，并且手牌总点数超过21，将ACE视为1
        if total > 21 and any(card.name == "ACE" for card in self.cards):
            total -= 10

        return total

    def discard_hand(self, deck: "Decks") -> None:
        """将手牌中的卡片放入弃牌堆中"""
        _len = len(self.cards)
        for _i in range(_len):
            if len(self.cards) == 0:
                raise ValueError("hand empty")
            deck.discard_pile.append(self.cards.pop())

# 定义牌堆
class Decks(NamedTuple):
    deck: List[Card]
    discard_pile: List[Card]

    @classmethod
    def new(cls) -> "Decks":
        """创建一个新的完整且洗牌过的牌堆，以及一个空的弃牌堆"""
        deck = Decks(deck=[], discard_pile=[])
        number_of_decks = 3

        # 填充牌堆
        for _n in range(number_of_decks):
            for card_name in CARD_NAMES:
                for _ in range(4):
                    deck.deck.append(Card(name=card_name))

        deck.shuffle()
        return deck

    def shuffle(self) -> None:
        """洗牌"""
        random.shuffle(self.deck)

    def draw_card(self) -> Card:
        """
        从牌堆中抽取一张卡片，并返回
        如果牌堆为空，则将弃牌堆洗入牌堆中，然后再次尝试
        """
        if len(self.deck) == 0:
            _len = len(self.discard_pile)

            if _len > 0:
                for _i in range(_len):
                    if len(self.discard_pile) == 0:
                        raise ValueError("discard pile is empty")
                    self.deck.append(self.discard_pile.pop())
                self.shuffle()
                return self.draw_card()
            else:
                raise Exception("discard pile empty")
        else:
            card = self.deck.pop()
            return card

# 定义卡片名称和起始余额
@dataclass
@dataclass
CARD_NAMES: List[str] = [
    "ACE",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "JACK",
    "QUEEN",
    "KING",
]
STARTING_BALANCE: int = 100

# 主函数
def main() -> None:
    game: Game

    print_welcome_screen()

    # 创建游戏
    game = Game.new(
        get_number_from_user_input("How many players should there be", 1, 7)
    )

    # 游戏循环，直到用户选择停止
    char = "y"
    while char == "y":
        game.play_game()
        char = get_char_from_user_input("Play Again?", ["y", "n"])

# 打印欢迎界面
def print_welcome_screen() -> None:
    print(
        """
                            BLACK JACK
              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY
    \n\n"""
    )

# 打印游戏说明
def print_instructions() -> None:
    print(
        """
    THIS IS THE GAME OF 21. AS MANY AS 7 PLAYERS MAY PLAY THE
    GAME. ON EACH DEAL, BETS WILL BE ASKED FOR, AND THE
    PLAYERS' BETS SHOULD BE TYPED IN. THE CARDS WILL THEN BE
    DEALT, AND EACH PLAYER IN TURN PLAYS HIS HAND. THE
    FIRST RESPONSE SHOULD BE EITHER 'D', INDICATING THAT THE
    PLAYER IS DOUBLING DOWN, 'S', INDICATING THAT HE IS
    STANDING, 'H', INDICATING HE WANTS ANOTHER CARD, OR '/',
    INDICATING THAT HE WANTS TO SPLIT HIS CARDS. AFTER THE
    INITIAL RESPONSE, ALL FURTHER RESPONSES SHOULD BE 'S' OR
    'H', UNLESS THE CARDS WERE SPLIT, IN WHICH CASE DOUBLING
    DOWN IS AGAIN PERMITTED. IN ORDER TO COLLECT FOR
    BLACKJACK, THE INITIAL RESPONSE SHOULD BE 'S'.
    NUMBER OF PLAYERS

    NOTE:'/' (splitting) is not currently implemented, and does nothing

    PRESS ENTER TO CONTINUE
    """
    )
    input()

# 从用户输入中获取一个整数
def get_number_from_user_input(prompt: str, min_value: int, max_value: int) -> int:
    user_input = None
    while user_input is None or user_input < min_value or user_input > max_value:
        raw_input = input(prompt + f" ({min_value}-{max_value})? ")

        try:
            user_input = int(raw_input)
            if user_input < min_value or user_input > max_value:
                print("Invalid input, please try again")
        except ValueError:
            print("Invalid input, please try again")
    return user_input

# 从用户输入中获取一个字符
def get_char_from_user_input(prompt: str, valid_results: List[str]) -> str:
    user_input = None
    while user_input not in valid_results:
        user_input = input(prompt + f" {valid_results}? ").lower()
        if user_input not in valid_results:
            print("Invalid input, please try again")
    assert user_input is not None
    return user_input

# 清空控制台输出
def clear() -> None:
    print("\x1b[2J\x1b[0;0H")


if __name__ == "__main__":
    main()

```