# `basic-computer-games\10_Blackjack\python\blackjack.py`

```
"""
Blackjack

Ported by Martin Thoma in 2022,
using the rust implementation of AnthonyMichaelTDM
"""

# 导入所需的库和模块
import enum
import random
from dataclasses import dataclass
from typing import List, NamedTuple

# 定义玩家类型的枚举
class PlayerType(enum.Enum):
    Player = "Player"
    Dealer = "Dealer"

# 定义玩法的枚举
class Play(enum.Enum):
    Stand = enum.auto()
    Hit = enum.auto()
    DoubleDown = enum.auto()
    Split = enum.auto()

# 定义卡牌的命名元组
class Card(NamedTuple):
    name: str

    @property
    def value(self) -> int:
        """
        returns the value associated with a card with the passed name
        return 0 if the passed card name doesn't exist
        """
        # 返回卡牌对应的点数，如果卡牌不存在则返回0
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

# 定义手牌的命名元组
class Hand(NamedTuple):
    cards: List[Card]

    def add_card(self, card: Card) -> None:
        """add a passed card to this hand"""
        # 向手牌中添加一张卡牌
        self.cards.append(card)

    def get_total(self) -> int:
        """returns the total points of the cards in this hand"""
        # 返回手牌中卡牌的总点数
        total: int = 0
        for card in self.cards:
            total += int(card.value)

        # if there is an ACE, and the hand would otherwise bust,
        # treat the ace like it's worth 1
        # 如果有一张ACE，并且手牌总点数超过21，将ACE的点数视为1
        if total > 21 and any(card.name == "ACE" for card in self.cards):
            total -= 10

        return total

    def discard_hand(self, deck: "Decks") -> None:
        """adds the cards in hand into the discard pile"""
        # 将手牌中的卡牌加入弃牌堆
        _len = len(self.cards)
        for _i in range(_len):
            if len(self.cards) == 0:
                raise ValueError("hand empty")
            deck.discard_pile.append(self.cards.pop())

# 定义牌堆的命名元组
class Decks(NamedTuple):
    deck: List[Card]
    discard_pile: List[Card]

    @classmethod
    def new(cls) -> "Decks":
        """creates a new full and shuffled deck, and an empty discard pile"""
        # 创建一个新的完整且洗过牌的牌组，以及一个空的弃牌堆
        deck = Decks(deck=[], discard_pile=[])
        number_of_decks = 3

        # fill deck
        for _n in range(number_of_decks):
            # 使用 number_of_decks 副牌填充牌组
            for card_name in CARD_NAMES:
                # 每种牌添加4张，总共一副牌有每种牌4张
                for _ in range(4):
                    deck.deck.append(Card(name=card_name))

        deck.shuffle()
        return deck

    def shuffle(self) -> None:
        """shuffles the deck"""
        # 洗牌
        random.shuffle(self.deck)

    def draw_card(self) -> Card:
        """
        draw card from deck, and return it
        if deck is empty, shuffles discard pile into it and tries again
        """
        if len(self.deck) == 0:
            _len = len(self.discard_pile)

            if _len > 0:
                # 牌组为空，将弃牌堆洗入牌组并重试
                print("deck is empty, shuffling")
                for _i in range(_len):
                    if len(self.discard_pile) == 0:
                        raise ValueError("discard pile is empty")
                    self.deck.append(self.discard_pile.pop())
                self.shuffle()
                return self.draw_card()
            else:
                # 弃牌堆和牌组都为空，不应该发生
                raise Exception("discard pile empty")
        else:
            card = self.deck.pop()
            return card
# 使用 dataclass 装饰器定义 Player 类
@dataclass
class Player:
    # 玩家手牌
    hand: Hand
    # 玩家余额
    balance: int
    # 玩家下注
    bet: int
    # 玩家赢得的局数
    wins: int
    # 玩家类型
    player_type: PlayerType
    # 玩家索引
    index: int

    # 类方法，创建一个新的玩家
    @classmethod
    def new(cls, player_type: PlayerType, index: int) -> "Player":
        """creates a new player of the given type"""
        return Player(
            hand=Hand(cards=[]),
            balance=STARTING_BALANCE,
            bet=0,
            wins=0,
            player_type=player_type,
            index=index,
        )

    # 获取玩家名称
    def get_name(self) -> str:
        return f"{self.player_type}{self.index}"

    # 获取玩家下注
    def get_bet(self) -> None:
        """gets a bet from the player"""
        if PlayerType.Player == self.player_type:
            if self.balance < 1:
                print(f"{self.get_name()} is out of money :(")
                self.bet = 0
            self.bet = get_number_from_user_input(
                f"{self.get_name()}\tWhat is your bet", 1, self.balance
            )

    # 返回玩家手牌的字符串表示
    def hand_as_string(self, hide_dealer: bool) -> str:
        """
        returns a string of the players hand

        if player is a dealer, returns the first card in the hand followed
        by *'s for every other card
        if player is a player, returns every card and the total
        """
        if not hide_dealer:
            s = ""
            for cards_in_hand in self.hand.cards[::-1]:
                s += f"{cards_in_hand.name}\t"
            s += f"total points = {self.hand.get_total()}"
            return s
        else:
            if self.player_type == PlayerType.Dealer:
                s = ""
                for c in self.hand.cards[1::-1]:
                    s += f"{c.name}\t"
                return s
            elif self.player_type == PlayerType.Player:
                s = ""
                for cards_in_hand in self.hand.cards[::-1]:
                    s += f"{cards_in_hand.name}\t"
                s += f"total points = {self.hand.get_total()}"
                return s
        raise Exception("This is unreachable")
    # 获取玩家的“出牌”动作
    def get_play(self) -> Play:
        """get the players 'play'"""
        # 根据玩家类型执行不同的操作：
        # 如果是庄家，使用算法确定出牌动作
        # 如果是玩家，向用户询问输入
        if self.player_type == PlayerType.Dealer:
            if self.hand.get_total() > 16:
                return Play.Stand
            else:
                return Play.Hit
        elif self.player_type == PlayerType.Player:
            valid_results: List[str]
            if len(self.hand.cards) > 2:
                # 如果手中的牌超过2张，至少进行了一轮操作，所以不允许分牌和加倍
                valid_results = ["s", "h"]
            else:
                valid_results = ["s", "h", "d", "/"]
            play = get_char_from_user_input("\tWhat is your play?", valid_results)
            if play == "s":
                return Play.Stand
            elif play == "h":
                return Play.Hit
            elif play == "d":
                return Play.DoubleDown
            elif play == "/":
                return Play.Split
            else:
                raise ValueError(f"got invalid character {play}")
        raise Exception("This is unreachable")
# 使用 dataclass 装饰器定义 Game 类
@dataclass
class Game:
    players: List[Player]  # last item in this is the dealer
    decks: Decks
    games_played: int

    # 类方法，创建新的 Game 对象
    @classmethod
    def new(cls, num_players: int) -> "Game":
        # 初始化玩家列表
        players: List[Player] = []

        # 添加庄家
        players.append(Player.new(PlayerType.Dealer, 0))
        # 创建人类玩家（至少一个）
        players.append(Player.new(PlayerType.Player, 1))
        # 创建指定数量的玩家
        for i in range(2, num_players):  # one less than num_players players
            players.append(Player.new(PlayerType.Player, i))

        # 如果用户想要获得游戏说明，则打印游戏说明
        if get_char_from_user_input("Do you want instructions", ["y", "n"]) == "y":
            print_instructions()
        print()

        # 返回新创建的 Game 对象
        return Game(players=players, decks=Decks.new(), games_played=0)

    # 打印每个玩家的得分
    def _print_stats(self) -> None:
        """prints the score of every player"""
        print(f"{self.stats_as_string()}")

    # 返回每个玩家的胜利次数、余额和下注的字符串
    def stats_as_string(self) -> str:
        """returns a string of the wins, balance, and bets of every player"""
        s = ""
        for p in self.players:
            # 格式化玩家统计信息的呈现方式
            if p.player_type == PlayerType.Dealer:
                s += f"{p.get_name()} Wins:\t{p.wins}\n"
            elif p.player_type == PlayerType.Player:
                s += f"{p.get_name()} "
                s += f"Wins:\t{p.wins}\t\t"
                s += f"Balance:\t{p.balance}\t\tBet\t{p.bet}\n"
        return f"Scores:\n{s}"

# 定义卡牌名称列表
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
# 定义初始余额
STARTING_BALANCE: int = 100

# 主函数
def main() -> None:
    game: Game

    # 打印欢迎界面
    print_welcome_screen()

    # 创建游戏
    game = Game.new(
        get_number_from_user_input("How many players should there be", 1, 7)
    )

    # 游戏循环，直到用户想要停止
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
    # 等待用户按下回车键继续
    input()


# 从用户输入中获取一个整数
def get_number_from_user_input(prompt: str, min_value: int, max_value: int) -> int:
    """gets a int integer from user input"""
    # 输入循环
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
    """returns the first character they type"""
    user_input = None
    # 当用户输入不在有效结果列表中时，循环执行以下操作
    while user_input not in valid_results:
        # 提示用户输入，并将输入转换为小写
        user_input = input(prompt + f" {valid_results}? ").lower()
        # 如果用户输入不在有效结果列表中，打印错误信息
        if user_input not in valid_results:
            print("Invalid input, please try again")
    # 断言用户输入不为空
    assert user_input is not None
    # 返回用户输入
    return user_input
# 定义一个清空标准输出的函数，不返回任何内容
def clear() -> None:
    """clear std out"""
    # 使用 ANSI 转义码清空终端屏幕
    print("\x1b[2J\x1b[0;0H")

# 如果当前脚本被直接执行，则调用 main 函数
if __name__ == "__main__":
    main()
```