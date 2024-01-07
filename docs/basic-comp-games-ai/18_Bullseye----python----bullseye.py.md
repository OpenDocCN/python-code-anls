# `basic-computer-games\18_Bullseye\python\bullseye.py`

```

# 导入随机模块和数据类模块
import random
from dataclasses import dataclass
from typing import List

# 定义玩家类
@dataclass
class Player:
    name: str
    score: int = 0

# 打印游戏介绍
def print_intro() -> None:
    # 打印游戏标题和介绍
    print(" " * 32 + "BULLSEYE")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n" * 3, end="")
    print("IN THIS GAME, UP TO 20 PLAYERS THROW DARTS AT A TARGET")
    print("WITH 10, 20, 30, AND 40 POINT ZONES.  THE OBJECTIVE IS")
    print("TO GET 200 POINTS.")
    print()
    # 打印投掷方式和得分概率
    print("THROW", end="")
    print(" " * 20 + "DESCRIPTION", end="")
    print(" " * 45 + "PROBABLE SCORE")
    print(" 1", end="")
    print(" " * 20 + "FAST OVERARM", end="")
    print(" " * 45 + "BULLSEYE OR COMPLETE MISS")
    print(" 2", end="")
    print(" " * 20 + "CONTROLLED OVERARM", end="")
    print(" " * 45 + "10, 20 OR 30 POINTS")
    print(" 3", end="")
    print(" " * 20 + "UNDERARM", end="")
    print(" " * 45 + "ANYTHING")
    print()

# 打印游戏结束信息
def print_outro(players: List[Player], winners: List[int]) -> None:
    print()
    print("WE HAVE A WINNER!!")
    print()
    # 打印获胜者的得分
    for winner in winners:
        print(f"{players[winner].name} SCORED {players[winner].score} POINTS.")
    print()
    print("THANKS FOR THE GAME.")

# 主函数
def main() -> None:
    # 打印游戏介绍
    print_intro()
    # 初始化玩家列表和获胜者列表
    players: List[Player] = []
    winners: List[int] = []  # will point to indices of player_names

    # 获取玩家数量
    nb_players = int(input("HOW MANY PLAYERS? "))
    # 循环添加玩家
    for _ in range(nb_players):
        player_name = input("NAME OF PLAYER #")
        players.append(Player(player_name))

    round_number = 0
    # 游戏循环
    while len(winners) == 0:
        round_number += 1
        print()
        print(f"ROUND {round_number}---------")
        # 遍历每个玩家
        for player in players:
            print()
            # 玩家投掷镖
            while True:
                throw = int(input(f"{player.name}'S THROW? "))
                if throw not in [1, 2, 3]:
                    print("INPUT 1, 2, OR 3!")
                else:
                    break
            # 根据投掷方式计算得分概率
            if throw == 1:
                probability_1 = 0.65
                probability_2 = 0.55
                probability_3 = 0.5
                probability_4 = 0.5
            elif throw == 2:
                probability_1 = 0.99
                probability_2 = 0.77
                probability_3 = 0.43
                probability_4 = 0.01
            elif throw == 3:
                probability_1 = 0.95
                probability_2 = 0.75
                probability_3 = 0.45
                probability_4 = 0.05
            # 随机生成投掷结果
            throwing_luck = random.random()
            # 根据投掷结果计算得分
            if throwing_luck >= probability_1:
                print("BULLSEYE!!  40 POINTS!")
                points = 40
            elif throwing_luck >= probability_2:
                print("30-POINT ZONE!")
                points = 30
            elif throwing_luck >= probability_3:
                print("20-POINT ZONE")
                points = 20
            elif throwing_luck >= probability_4:
                print("WHEW!  10 POINTS.")
                points = 10
            else:
                print("MISSED THE TARGET!  TOO BAD.")
                points = 0
            # 更新玩家得分
            player.score += points
            print(f"TOTAL SCORE = {player.score}")
        # 判断是否有玩家获胜
        for player_index, player in enumerate(players):
            if player.score > 200:
                winners.append(player_index)

    # 打印游戏结束信息
    print_outro(players, winners)

# 程序入口
if __name__ == "__main__":
    main()

```