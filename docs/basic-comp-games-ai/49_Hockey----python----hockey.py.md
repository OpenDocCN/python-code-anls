# `basic-computer-games\49_Hockey\python\hockey.py`

```

"""
HOCKEY

A simulation of an ice hockey game.

The original author is Robert Puopolo;
modifications by Steve North of Creative Computing.

Ported to Python by Martin Thoma in 2022
"""

from dataclasses import dataclass, field  # 导入 dataclass 模块中的 dataclass 和 field 函数
from random import randint  # 从 random 模块中导入 randint 函数
from typing import List, Tuple  # 从 typing 模块中导入 List 和 Tuple 类型

NB_PLAYERS = 6  # 定义常量 NB_PLAYERS 为 6


@dataclass  # 使用 dataclass 装饰器
class Team:  # 定义 Team 类
    # TODO: It would be better to use a Player-class (name, goals, assits)
    #       and have the attributes directly at each player. This would avoid
    #       dealing with indices that much
    #
    #       I'm also rather certain that I messed up somewhere with the indices
    #       - instead of using those, one could use actual player positions:
    #       LEFT WING,    CENTER,        RIGHT WING
    #       LEFT DEFENSE, RIGHT DEFENSE, GOALKEEPER
    name: str  # 定义属性 name 为字符串类型
    players: List[str]  # 定义属性 players 为字符串类型的列表，长度为 6
    shots_on_net: int = 0  # 定义属性 shots_on_net 为整数类型，默认值为 0
    goals: List[int] = field(default_factory=lambda: [0 for _ in range(NB_PLAYERS)])  # 定义属性 goals 为整数类型的列表，长度为 6，初始值为 0
    assists: List[int] = field(default_factory=lambda: [0 for _ in range(NB_PLAYERS)])  # 定义属性 assists 为整数类型的列表，长度为 6，初始值为 0
    score: int = 0  # 定义属性 score 为整数类型，默认值为 0

    def show_lineup(self) -> None:  # 定义方法 show_lineup，返回类型为 None
        print(" " * 10 + f"{self.name} STARTING LINEUP")  # 打印球队名称和 STARTING LINEUP
        for player in self.players:  # 遍历球队的球员
            print(player)  # 打印球员的名字


def ask_binary(prompt: str, error_msg: str) -> bool:  # 定义函数 ask_binary，参数为 prompt 和 error_msg，返回类型为布尔值
    while True:  # 进入循环
        answer = input(prompt).lower()  # 获取用户输入并转换为小写
        if answer in ["y", "yes"]:  # 如果用户输入为 "y" 或 "yes"
            return True  # 返回 True
        if answer in ["n", "no"]:  # 如果用户输入为 "n" 或 "no"
            return False  # 返回 False
        print(error_msg)  # 打印错误消息


# 其余部分的注释请参考上述示例的注释方式，对代码中的函数、类、变量等进行解释

```