# `18_Bullseye\python\bullseye.py`

```
import random  # 导入 random 模块，用于生成随机数
from dataclasses import dataclass  # 导入 dataclass 类装饰器，用于创建数据类
from typing import List  # 导入 List 类型提示，用于声明列表类型


@dataclass  # 使用 dataclass 装饰器创建数据类
class Player:  # 定义 Player 类
    name: str  # 定义 name 属性为字符串类型
    score: int = 0  # 定义 score 属性为整数类型，默认值为 0


def print_intro() -> None:  # 定义函数 print_intro，返回类型为 None
    print(" " * 32 + "BULLSEYE")  # 打印字符串 "BULLSEYE"
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 打印字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    print("\n" * 3, end="")  # 打印三个换行符
    print("IN THIS GAME, UP TO 20 PLAYERS THROW DARTS AT A TARGET")  # 打印游戏介绍
    print("WITH 10, 20, 30, AND 40 POINT ZONES.  THE OBJECTIVE IS")  # 打印游戏目标
    print("TO GET 200 POINTS.")  # 打印游戏目标
    print()  # 打印空行
    print("THROW", end="")  # 打印字符串 "THROW"
    # 打印描述和可能的得分的表头
    print(" " * 20 + "DESCRIPTION", end="")
    print(" " * 45 + "PROBABLE SCORE")
    # 打印不同投掷方式的编号、描述和可能的得分
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

def print_outro(players: List[Player], winners: List[int]) -> None:
    # 打印游戏结束的提示
    print()
    print("WE HAVE A WINNER!!")
    print()
    # 遍历获胜者列表，打印他们的名字和得分
    for winner in winners:
        print(f"{players[winner].name} SCORED {players[winner].score} POINTS.")
    print()  # 打印空行
    print("THANKS FOR THE GAME.")  # 打印感谢游戏信息


def main() -> None:
    print_intro()  # 调用打印游戏介绍的函数

    players: List[Player] = []  # 创建一个空的玩家列表

    winners: List[int] = []  # 创建一个空的获胜者列表，将指向玩家名称的索引

    nb_players = int(input("HOW MANY PLAYERS? "))  # 获取玩家数量
    for _ in range(nb_players):  # 循环获取每个玩家的名称
        player_name = input("NAME OF PLAYER #")
        players.append(Player(player_name))  # 将玩家名称添加到玩家列表中

    round_number = 0  # 初始化回合数为0
    while len(winners) == 0:  # 当获胜者列表为空时循环
        round_number += 1  # 回合数加1
        print()  # 打印空行
        print(f"ROUND {round_number}---------")  # 打印当前回合数
        for player in players:  # 遍历玩家列表
            print()  # 打印空行
            while True:  # 进入无限循环
                throw = int(input(f"{player.name}'S THROW? "))  # 获取玩家输入的投掷值
                if throw not in [1, 2, 3]:  # 如果投掷值不在1、2、3中
                    print("INPUT 1, 2, OR 3!")  # 打印提示信息
                else:  # 如果投掷值在1、2、3中
                    break  # 退出循环
            if throw == 1:  # 如果投掷值为1
                probability_1 = 0.65  # 设置概率1为0.65
                probability_2 = 0.55  # 设置概率2为0.55
                probability_3 = 0.5  # 设置概率3为0.5
                probability_4 = 0.5  # 设置概率4为0.5
            elif throw == 2:  # 如果投掷值为2
                probability_1 = 0.99  # 设置概率1为0.99
                probability_2 = 0.77  # 设置概率2为0.77
                probability_3 = 0.43  # 设置概率3为0.43
                probability_4 = 0.01  # 设置概率4为0.01
            elif throw == 3:  # 如果投掷值为3
                probability_1 = 0.95  # 设置概率1为0.95
                probability_2 = 0.75  # 设置命中30分区域的概率
                probability_3 = 0.45  # 设置命中20分区域的概率
                probability_4 = 0.05  # 设置命中10分区域的概率
            throwing_luck = random.random()  # 生成一个随机数，表示投掷的运气
            if throwing_luck >= probability_1:  # 如果随机数大于等于命中40分区域的概率
                print("BULLSEYE!!  40 POINTS!")  # 打印命中40分区域的提示
                points = 40  # 设置得分为40分
            elif throwing_luck >= probability_2:  # 如果随机数大于等于命中30分区域的概率
                print("30-POINT ZONE!")  # 打印命中30分区域的提示
                points = 30  # 设置得分为30分
            elif throwing_luck >= probability_3:  # 如果随机数大于等于命中20分区域的概率
                print("20-POINT ZONE")  # 打印命中20分区域的提示
                points = 20  # 设置得分为20分
            elif throwing_luck >= probability_4:  # 如果随机数大于等于命中10分区域的概率
                print("WHEW!  10 POINTS.")  # 打印命中10分区域的提示
                points = 10  # 设置得分为10分
            else:  # 如果随机数小于所有命中区域的概率
                print("MISSED THE TARGET!  TOO BAD.")  # 打印未命中目标的提示
                points = 0  # 设置得分为0分
            player.score += points  # 将得分加到玩家的总分上
        print(f"TOTAL SCORE = {player.score}")  # 打印每个玩家的总分数
    for player_index, player in enumerate(players):  # 使用enumerate函数遍历玩家列表，获取玩家索引和玩家对象
        if player.score > 200:  # 如果玩家的分数大于200
            winners.append(player_index)  # 将该玩家的索引添加到获胜者列表中

print_outro(players, winners)  # 调用print_outro函数，传入玩家列表和获胜者列表作为参数


if __name__ == "__main__":  # 如果当前脚本被直接执行
    main()  # 调用main函数
```