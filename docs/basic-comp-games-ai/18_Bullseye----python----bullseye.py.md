# `basic-computer-games\18_Bullseye\python\bullseye.py`

```
# 导入 random 模块
import random
# 从 dataclasses 模块中导入 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块中导入 List 类型
from typing import List

# 使用 dataclass 装饰器创建 Player 类，包含 name 和 score 两个属性
@dataclass
class Player:
    name: str
    score: int = 0

# 定义函数 print_intro，无返回值
def print_intro() -> None:
    # 打印游戏标题
    print(" " * 32 + "BULLSEYE")
    # 打印游戏信息
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n" * 3, end="")
    print("IN THIS GAME, UP TO 20 PLAYERS THROW DARTS AT A TARGET")
    print("WITH 10, 20, 30, AND 40 POINT ZONES.  THE OBJECTIVE IS")
    print("TO GET 200 POINTS.")
    print()
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

# 定义函数 print_outro，接受 players 和 winners 两个参数，无返回值
def print_outro(players: List[Player], winners: List[int]) -> None:
    # 打印游戏结束信息
    print()
    print("WE HAVE A WINNER!!")
    print()
    # 遍历获胜者列表，打印获胜者的名字和分数
    for winner in winners:
        print(f"{players[winner].name} SCORED {players[winner].score} POINTS.")
    print()
    print("THANKS FOR THE GAME.")

# 定义主函数 main，无返回值
def main() -> None:
    # 打印游戏介绍
    print_intro()
    # 初始化玩家列表为空
    players: List[Player] = []

    # 初始化获胜者列表为空
    winners: List[int] = []  # will point to indices of player_names

    # 获取玩家数量
    nb_players = int(input("HOW MANY PLAYERS? "))
    # 循环获取每个玩家的名字并添加到玩家列表中
    for _ in range(nb_players):
        player_name = input("NAME OF PLAYER #")
        players.append(Player(player_name))

    # 初始化回合数为 0
    round_number = 0
    # 当获胜者数量为0时，执行循环
    while len(winners) == 0:
        # 回合数加1
        round_number += 1
        # 打印回合数
        print()
        print(f"ROUND {round_number}---------")
        # 遍历玩家列表
        for player in players:
            # 打印空行
            print()
            # 循环直到输入合法的投掷值
            while True:
                throw = int(input(f"{player.name}'S THROW? "))
                if throw not in [1, 2, 3]:
                    print("INPUT 1, 2, OR 3!")
                else:
                    break
            # 根据投掷值设置不同的概率
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
            # 生成一个随机数，表示投掷的运气
            throwing_luck = random.random()
            # 根据投掷的运气和概率判断得分
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
            # 累加玩家得分
            player.score += points
            # 打印玩家总得分
            print(f"TOTAL SCORE = {player.score}")
        # 遍历玩家列表，检查是否有得分超过200的玩家
        for player_index, player in enumerate(players):
            if player.score > 200:
                # 将得分超过200的玩家索引添加到获胜者列表中
                winners.append(player_index)

    # 打印游戏结束信息
    print_outro(players, winners)
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```