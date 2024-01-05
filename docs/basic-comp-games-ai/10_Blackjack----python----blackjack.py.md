# `10_Blackjack\python\blackjack.py`

```
"""
Blackjack

Ported by Martin Thoma in 2022,
using the rust implementation of AnthonyMichaelTDM
"""

import enum  # 导入枚举模块
import random  # 导入随机数模块
from dataclasses import dataclass  # 导入数据类模块
from typing import List, NamedTuple  # 导入类型提示模块


class PlayerType(enum.Enum):  # 定义玩家类型枚举类
    Player = "Player"  # 玩家
    Dealer = "Dealer"  # 庄家


class Play(enum.Enum):  # 定义玩法枚举类
    Stand = enum.auto()  # 站立
    Hit = enum.auto()  # 创建一个名为Hit的枚举成员，其值为自动分配的下一个整数
    DoubleDown = enum.auto()  # 创建一个名为DoubleDown的枚举成员，其值为自动分配的下一个整数
    Split = enum.auto()  # 创建一个名为Split的枚举成员，其值为自动分配的下一个整数


class Card(NamedTuple):  # 定义一个名为Card的命名元组
    name: str  # 定义一个名为name的属性，类型为字符串

    @property  # 将下面的value方法定义为属性
    def value(self) -> int:  # 定义一个名为value的方法，返回类型为整数
        """
        returns the value associated with a card with the passed name
        return 0 if the passed card name doesn't exist
        """
        return {  # 返回一个字典
            "ACE": 11,  # 键为"ACE"，值为11
            "2": 2,  # 键为"2"，值为2
            "3": 3,  # 键为"3"，值为3
            "4": 4,  # 键为"4"，值为4
            "5": 5,  # 键为"5"，值为5
            "6": 6,  # 将字符串"6"映射到整数6
            "7": 7,  # 将字符串"7"映射到整数7
            "8": 8,  # 将字符串"8"映射到整数8
            "9": 9,  # 将字符串"9"映射到整数9
            "10": 10,  # 将字符串"10"映射到整数10
            "JACK": 10,  # 将字符串"JACK"映射到整数10
            "QUEEN": 10,  # 将字符串"QUEEN"映射到整数10
            "KING": 10,  # 将字符串"KING"映射到整数10
        }.get(self.name, 0)  # 如果self.name在字典中，则返回对应的值，否则返回0


class Hand(NamedTuple):
    cards: List[Card]

    def add_card(self, card: Card) -> None:
        """add a passed card to this hand"""
        self.cards.append(card)  # 将传入的卡片添加到手牌列表中

    def get_total(self) -> int:
        """returns the total points of the cards in this hand"""
        total: int = 0  # 初始化变量total，用于存储手中牌的总点数
        for card in self.cards:  # 遍历手中的每张牌
            total += int(card.value)  # 将每张牌的点数加到total中

        # 如果手中有ACE，并且总点数超过21，将ACE的点数视为1
        if total > 21 and any(card.name == "ACE" for card in self.cards):
            total -= 10

        return total  # 返回手中牌的总点数

    def discard_hand(self, deck: "Decks") -> None:
        """将手中的牌加入弃牌堆中"""
        _len = len(self.cards)  # 获取手中牌的数量
        for _i in range(_len):  # 遍历手中的每张牌
            if len(self.cards) == 0:  # 如果手中没有牌了
                raise ValueError("hand empty")  # 抛出数值错误
            deck.discard_pile.append(self.cards.pop())  # 将手中的牌加入弃牌堆中
class Decks(NamedTuple):
    deck: List[Card]  # 用于存储牌堆的列表
    discard_pile: List[Card]  # 用于存储弃牌堆的列表

    @classmethod
    def new(cls) -> "Decks":
        """creates a new full and shuffled deck, and an empty discard pile"""
        # 创建一个新的牌堆对象，包括一个完整且洗过的牌堆，以及一个空的弃牌堆
        deck = Decks(deck=[], discard_pile=[])
        number_of_decks = 3  # 设置牌堆的数量为3副牌

        # fill deck
        for _n in range(number_of_decks):
            # 填充牌堆
            for card_name in CARD_NAMES:  # 遍历所有的牌名
                # 添加每种牌的4张，总共构成一副牌，每种牌有4张
                for _ in range(4):
                    deck.deck.append(Card(name=card_name))  # 将牌添加到牌堆中

        deck.shuffle()  # 洗牌
        return deck  # 返回牌组

    def shuffle(self) -> None:
        """shuffles the deck"""  # 洗牌
        random.shuffle(self.deck)  # 使用随机函数对牌组进行洗牌

    def draw_card(self) -> Card:
        """
        draw card from deck, and return it
        if deck is empty, shuffles discard pile into it and tries again
        """  # 从牌组中抽取一张牌并返回
        if len(self.deck) == 0:  # 如果牌组为空
            _len = len(self.discard_pile)  # 获取弃牌堆的长度

            if _len > 0:  # 如果弃牌堆不为空
                # deck is empty, shuffle discard pile into deck and try again
                print("deck is empty, shuffling")  # 输出信息：牌组为空，正在洗牌
                for _i in range(_len):  # 遍历弃牌堆
                    if len(self.discard_pile) == 0:  # 如果弃牌堆为空
                        raise ValueError("discard pile is empty")  # 抛出数值错误：弃牌堆为空
                self.deck.append(self.discard_pile.pop())  # 将弃牌堆顶部的牌加入到牌堆中
                self.shuffle()  # 洗牌
                return self.draw_card()  # 抽一张牌
            else:
                # 弃牌堆和牌堆都为空，不应该发生
                raise Exception("discard pile empty")  # 抛出异常，表示弃牌堆为空
        else:
            card = self.deck.pop()  # 从牌堆中移除一张牌
            return card  # 返回移除的牌


@dataclass
class Player:
    hand: Hand  # 玩家手中的牌
    balance: int  # 玩家的余额
    bet: int  # 玩家的赌注
    wins: int  # 玩家的胜利次数
    player_type: PlayerType  # 玩家类型
    index: int  # 玩家的索引
    @classmethod
    def new(cls, player_type: PlayerType, index: int) -> "Player":
        """创建给定类型的新玩家"""
        return Player(
            hand=Hand(cards=[]),
            balance=STARTING_BALANCE,
            bet=0,
            wins=0,
            player_type=player_type,
            index=index,
        )

    def get_name(self) -> str:
        return f"{self.player_type}{self.index}"

    def get_bet(self) -> None:
        """从玩家获取赌注"""
        if PlayerType.Player == self.player_type:
            if self.balance < 1:
                print(f"{self.get_name()} is out of money :(")
                self.bet = 0  # 初始化赌注为0
            self.bet = get_number_from_user_input(  # 从用户输入中获取赌注
                f"{self.get_name()}\tWhat is your bet", 1, self.balance
            )

    def hand_as_string(self, hide_dealer: bool) -> str:
        """
        返回玩家手中的牌的字符串表示

        如果玩家是庄家，返回手中的第一张牌，后面跟着*号代表其他的牌
        如果玩家是玩家，返回每张牌和总点数
        """
        if not hide_dealer:  # 如果不隐藏庄家的牌
            s = ""  # 初始化字符串为空
            for cards_in_hand in self.hand.cards[::-1]:  # 遍历玩家手中的牌
                s += f"{cards_in_hand.name}\t"  # 将每张牌的名称添加到字符串中
            s += f"total points = {self.hand.get_total()}"  # 添加总点数到字符串中
            return s  # 返回字符串
        else:  # 如果隐藏庄家的牌
            # 如果玩家类型是庄家
            if self.player_type == PlayerType.Dealer:
                # 初始化一个空字符串
                s = ""
                # 遍历庄家手中的牌，从第二张牌开始到第一张牌结束
                for c in self.hand.cards[1::-1]:
                    # 将每张牌的名称添加到字符串中，用制表符分隔
                    s += f"{c.name}\t"
                # 返回拼接好的字符串
                return s
            # 如果玩家类型是玩家
            elif self.player_type == PlayerType.Player:
                # 初始化一个空字符串
                s = ""
                # 遍历玩家手中的牌，从最后一张牌到第一张牌结束
                for cards_in_hand in self.hand.cards[::-1]:
                    # 将每张牌的名称添加到字符串中，用制表符分隔
                    s += f"{cards_in_hand.name}\t"
                # 将玩家手中牌的总点数添加到字符串末尾
                s += f"total points = {self.hand.get_total()}"
                # 返回拼接好的字符串
                return s
        # 如果以上条件都不满足，抛出异常
        raise Exception("This is unreachable")

    def get_play(self) -> Play:
        """获取玩家的出牌"""
        # 根据玩家类型执行不同的操作：
        # 如果是庄家，使用算法确定出牌
        # 如果是玩家，向用户询问出牌
        if self.player_type == PlayerType.Dealer:
            # 如果庄家手中的牌总点数大于16
            return Play.Split
```

注释：

- `valid_results: List[str]`：声明一个变量valid_results，类型为字符串列表，用于存储有效的玩家操作结果
- `if len(self.hand.cards) > 2:`：如果玩家手中的牌数量大于2
- `valid_results = ["s", "h"]`：将有效的玩家操作结果设置为["s", "h"]，即站牌和要牌
- `else:`：否则
- `valid_results = ["s", "h", "d", "/"]`：将有效的玩家操作结果设置为["s", "h", "d", "/"]，即站牌、要牌、加倍和分牌
- `play = get_char_from_user_input("\tWhat is your play?", valid_results)`：通过用户输入获取玩家的操作结果，限定在valid_results中
- `if play == "s":`：如果玩家选择站牌
- `return Play.Stand`：返回站牌操作
- `elif play == "h":`：否则，如果玩家选择要牌
- `return Play.Hit`：返回要牌操作
- `elif play == "d":`：否则，如果玩家选择加倍
- `return Play.DoubleDown`：返回加倍操作
- `elif play == "/":`：否则，如果玩家选择分牌
- `return Play.Split`：返回分牌操作
                return Play.Split  # 返回分牌操作
            else:
                raise ValueError(f"got invalid character {play}")  # 如果play不是有效的操作，抛出值错误异常
        raise Exception("This is unreachable")  # 如果程序执行到这里，抛出异常，表示不可达的代码


@dataclass
class Game:
    players: List[Player]  # 玩家列表，最后一个是庄家
    decks: Decks  # 牌组
    games_played: int  # 已玩游戏数

    @classmethod
    def new(cls, num_players: int) -> "Game":  # 类方法，用于创建新的游戏实例
        players: List[Player] = []  # 初始化玩家列表

        # 添加庄家
        players.append(Player.new(PlayerType.Dealer, 0))
        # 创建人类玩家（至少一个）
        players.append(Player.new(PlayerType.Player, 1))
        for i in range(2, num_players):  # one less than num_players players
            # 循环创建指定数量的玩家对象并添加到players列表中
            players.append(Player.new(PlayerType.Player, i))

        if get_char_from_user_input("Do you want instructions", ["y", "n"]) == "y":
            # 如果用户输入'y'，则打印游戏说明
            print_instructions()
        print()

        return Game(players=players, decks=Decks.new(), games_played=0)
        # 返回一个包含玩家、牌组和游戏次数的Game对象

    def _print_stats(self) -> None:
        """prints the score of every player"""
        # 打印每个玩家的得分
        print(f"{self.stats_as_string()}")

    def stats_as_string(self) -> str:
        """returns a string of the wins, balance, and bets of every player"""
        # 返回包含每个玩家胜利次数、余额和下注的字符串
        s = ""
        for p in self.players:
            # 遍历每个玩家
            if p.player_type == PlayerType.Dealer:
                # 如果玩家类型是Dealer，则格式化玩家统计信息并添加到s中
                s += f"{p.get_name()} Wins:\t{p.wins}\n"
            elif p.player_type == PlayerType.Player:  # 如果玩家类型是玩家
                s += f"{p.get_name()} "  # 将玩家的名字添加到字符串s中
                s += f"Wins:\t{p.wins}\t\t"  # 将玩家的胜利次数添加到字符串s中
                s += f"Balance:\t{p.balance}\t\tBet\t{p.bet}\n"  # 将玩家的余额和下注金额添加到字符串s中
        return f"Scores:\n{s}"  # 返回包含玩家得分信息的字符串

    def play_game(self) -> None:
        """plays a round of blackjack"""  # 玩一轮二十一点游戏
        game = self.games_played  # 获取已玩游戏的数量
        player_hands_message: str = ""  # 初始化玩家手牌信息的字符串为空

        # deal two cards to each player  # 给每个玩家发两张牌
        for _i in range(2):  # 循环两次
            for player in self.players:  # 遍历每个玩家
                player.hand.add_card(self.decks.draw_card())  # 给每个玩家的手牌添加一张牌

        # get everyones bets  # 获取每个玩家的下注金额
        for player in self.players:  # 遍历每个玩家
            player.get_bet()  # 获取玩家的下注金额
        scores = self.stats_as_string()  # 获取游戏统计信息并存储在scores变量中
        # 对每个玩家进行游戏
        for player in self.players:
            # 回合循环，直到玩家完成回合结束
            while True:
                clear()  # 清空屏幕
                print_welcome_screen()  # 打印欢迎界面
                print(f"\n\t\t\tGame {game}")  # 打印游戏编号
                print(scores)  # 打印分数
                print(player_hands_message)  # 打印玩家手牌信息
                print(f"{player.get_name()} Hand:\t{player.hand_as_string(True)}")  # 打印玩家手牌

                if PlayerType.Player == player.player_type and player.bet == 0:  # 如果是玩家类型且下注为0，则跳出循环
                    break

                # 进行回合
                # 检查他们的手牌价值是否为21点或爆牌
                score = player.hand.get_total()  # 获取玩家手牌总点数
                if score >= 21:  # 如果点数大于等于21
                    if score == 21:  # 如果点数等于21
                # get player move
                # 获取玩家的动作
                play = player.get_play()
                # process play
                # 处理玩家的动作
                if play == Play.Stand:
                    # 如果玩家选择站立
                    print(f"\t{play}")
                    break
                elif play == Play.Hit:
                    # 如果玩家选择要牌
                    print(f"\t{play}")
                    player.hand.add_card(self.decks.draw_card())
                elif play == Play.DoubleDown:
                    # 如果玩家选择加倍
                    print(f"\t{play}")

                    # double their balance if there's enough money,
                    # othewise go all-in
                    # 如果有足够的钱，将他们的赌注加倍，否则全下
                    if player.bet * 2 < player.balance:
                        # 如果玩家的赌注乘以2小于玩家的余额
                        player.bet *= 2  # 如果玩家选择加倍下注，将下注金额翻倍
                    else:
                        player.bet = player.balance  # 如果玩家选择不加倍下注，下注金额等于玩家余额
                    player.hand.add_card(self.decks.draw_card())  # 玩家抽取一张牌加入手牌
                elif play == Play.Split:  # 如果玩家选择分牌
                    pass  # 什么都不做

            # 将玩家的手牌信息添加到消息中
            player_hands_message += (
                f"{player.get_name()} Hand:\t{player.hand_as_string(True)}\n"
            )

        # 确定赢家
        top_score = 0  # 初始化最高分为0

        # 获得最高分的玩家数量
        num_winners = 1  # 初始化最高分玩家数量为1

        non_burst_players = [  # 获取没有爆牌的玩家列表
            player for player in self.players if player.hand.get_total() <= 21
        # 遍历非爆牌玩家列表
        for player in non_burst_players:
            # 获取玩家手牌的总分
            score = player.hand.get_total()
            # 如果分数大于最高分
            if score > top_score:
                # 更新最高分和赢家数量
                top_score = score
                num_winners = 1
            # 如果分数等于最高分
            elif score == top_score:
                # 增加赢家数量
                num_winners += 1

        # 展示赢家
        top_score_players = [
            player
            for player in non_burst_players
            if player.hand.get_total() == top_score
        ]
        # 遍历最高分玩家列表
        for x in top_score_players:
            # 打印赢家的名字
            print(f"{x.get_name()} ")
            # 增加他们的胜利次数
            x.wins += 1
        # 如果赢家数量大于1
        if num_winners > 1:
        # 打印所有得分与最高分相同的玩家
        print(f"all tie with {top_score}\n\n\n")
        # 如果有玩家得分最高，则打印获胜信息
        else:
            print(
                f"wins with {top_score}!\n\n\n",
            )

        # 处理赌注
        # 从输家账户中扣除赌注
        losers = [
            player for player in self.players if player.hand.get_total() != top_score
        ]
        for loser in losers:
            loser.balance -= loser.bet
        # 将赌注添加到赢家账户中
        winners = [
            player for player in self.players if player.hand.get_total() == top_score
        ]
        for winner in winners:
            winner.balance += winner.bet
        # discard hands
        # 丢弃玩家手中的牌
        for player in self.players:
            player.hand.discard_hand(self.decks)

        # increment games_played
        # 增加游戏次数
        self.games_played += 1


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
```

在这个示例中，第一段代码是一个类中的方法，它首先遍历玩家列表，然后调用每个玩家的手牌对象的discard_hand方法来丢弃手中的牌。接着，它增加了游戏次数。

第二段代码定义了一个名为CARD_NAMES的字符串列表，其中包含了扑克牌的名称。
    "QUEEN",  # 添加"QUEEN"到常量列表中
    "KING",   # 添加"KING"到常量列表中
]
STARTING_BALANCE: int = 100  # 设置初始余额为100


def main() -> None:  # 定义主函数，不返回任何结果
    game: Game  # 声明game变量为Game类型

    print_welcome_screen()  # 调用打印欢迎界面的函数

    # create game
    game = Game.new(  # 创建游戏对象
        get_number_from_user_input("How many players should there be", 1, 7)  # 从用户输入获取玩家数量
    )

    # game loop, play game until user wants to stop
    char = "y"  # 初始化char变量为"y"
    while char == "y":  # 当char为"y"时循环执行下面的代码
        game.play_game()  # 调用游戏进行游戏
        char = get_char_from_user_input("Play Again?", ["y", "n"])  # 从用户输入中获取字符，只能是"y"或者"n"

def print_welcome_screen() -> None:  # 打印欢迎界面
    print(
        """
                            BLACK JACK
              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY
    \n\n"""
    )


def print_instructions() -> None:  # 打印游戏说明
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
    # 这段代码是一段注释，解释了玩家在游戏中的不同操作对应的指令，以及初始响应应该是什么。

    NOTE:'/' (splitting) is not currently implemented, and does nothing
    # 这段代码是一段注释，解释了当前游戏中的一个功能（分牌）尚未实现，输入 '/' 指令没有任何效果。

    PRESS ENTER TO CONTINUE
    """
    )
    input()
    # 提示用户按回车键继续游戏，并等待用户输入

def get_number_from_user_input(prompt: str, min_value: int, max_value: int) -> int:
    """gets a int integer from user input"""
    # input loop
    # 这段代码是一个函数定义，用于从用户输入中获取一个整数值，包括了提示信息、最小值和最大值的限制。
    user_input = None  # 初始化用户输入变量为 None
    while user_input is None or user_input < min_value or user_input > max_value:  # 当用户输入为 None 或者小于最小值或者大于最大值时循环
        raw_input = input(prompt + f" ({min_value}-{max_value})? ")  # 提示用户输入，并将输入存储在 raw_input 变量中

        try:  # 尝试将用户输入转换为整数
            user_input = int(raw_input)  # 将用户输入转换为整数并存储在 user_input 变量中
            if user_input < min_value or user_input > max_value:  # 如果用户输入小于最小值或者大于最大值
                print("Invalid input, please try again")  # 打印错误提示信息
        except ValueError:  # 如果用户输入无法转换为整数
            print("Invalid input, please try again")  # 打印错误提示信息
    return user_input  # 返回用户输入的整数值


def get_char_from_user_input(prompt: str, valid_results: List[str]) -> str:
    """returns the first character they type"""  # 函数说明文档
    user_input = None  # 初始化用户输入变量为 None
    while user_input not in valid_results:  # 当用户输入不在有效结果列表中时循环
        user_input = input(prompt + f" {valid_results}? ").lower()  # 提示用户输入，并将输入转换为小写存储在 user_input 变量中
        if user_input not in valid_results:  # 如果用户输入不在有效结果列表中
            print("Invalid input, please try again")  # 打印错误提示信息
    assert user_input is not None  # 断言用户输入不为空
    return user_input  # 返回用户输入的值


def clear() -> None:
    """clear std out"""  # 清空标准输出
    print("\x1b[2J\x1b[0;0H")  # 使用 ANSI 转义序列清空终端屏幕


if __name__ == "__main__":
    main()  # 如果作为主程序运行，则调用 main 函数
```