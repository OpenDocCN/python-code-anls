# `basic-computer-games\15_Boxing\python\boxing.py`

```

#!/usr/bin/env python3
# 指定使用 Python3 解释器

import json
# 导入 json 模块，用于处理 JSON 数据
import random
# 导入 random 模块，用于生成随机数
from dataclasses import dataclass
# 从 dataclasses 模块中导入 dataclass 装饰器，用于创建数据类
from pathlib import Path
# 导入 Path 类，用于处理文件路径
from typing import Dict, Literal, NamedTuple, Tuple
# 从 typing 模块中导入 Dict、Literal、NamedTuple 和 Tuple 类型，用于类型提示

class PunchProfile(NamedTuple):
    # 定义 PunchProfile 类，继承自 NamedTuple
    choices: int
    threshold: int
    hit_damage: int
    block_damage: int
    pre_msg: str
    hit_msg: str
    blocked_msg: str
    knockout_possible: bool = False
    # 定义 PunchProfile 类的属性和方法

    def is_hit(self) -> bool:
        # 定义 is_hit 方法，返回布尔值
        return random.randint(1, self.choices) <= self.threshold

@dataclass
# 使用 dataclass 装饰器定义 Player 类
class Player:
    name: str
    best: int  # this hit guarantees 2 damage on opponent
    weakness: int  # you're always hit when your opponent uses this punch
    is_computer: bool
    punch_profiles: Dict[Literal[1, 2, 3, 4], PunchProfile]
    damage: int = 0
    score: int = 0
    knockedout: bool = False
    # 定义 Player 类的属性和方法

    def get_punch_choice(self) -> Literal[1, 2, 3, 4]:
        # 定义 get_punch_choice 方法，返回 Literal[1, 2, 3, 4] 类型
        if self.is_computer:
            return random.randint(1, 4)  # type: ignore
        else:
            punch = -1
            while punch not in [1, 2, 3, 4]:
                print(f"{self.name}'S PUNCH", end="? ")
                punch = int(input())
            return punch  # type: ignore

KNOCKOUT_THRESHOLD = 35
# 定义 KNOCKOUT_THRESHOLD 常量

QUESTION_PROMPT = "? "
# 定义 QUESTION_PROMPT 常量
KNOCKED_COLD = "{loser} IS KNOCKED COLD AND {winner} IS THE WINNER AND CHAMP"
# 定义 KNOCKED_COLD 常量

def get_vulnerability() -> int:
    # 定义 get_vulnerability 函数，返回整数
    print("WHAT IS HIS VULNERABILITY", end=QUESTION_PROMPT)
    vulnerability = int(input())
    return vulnerability

def get_opponent_stats() -> Tuple[int, int]:
    # 定义 get_opponent_stats 函数，返回元组
    opponent_best = 0
    opponent_weakness = 0
    while opponent_best == opponent_weakness:
        opponent_best = random.randint(1, 4)
        opponent_weakness = random.randint(1, 4)
    return opponent_best, opponent_weakness

def read_punch_profiles(filepath: Path) -> Dict[Literal[1, 2, 3, 4], PunchProfile]:
    # 定义 read_punch_profiles 函数，接收文件路径参数，返回 PunchProfile 字典
    with open(filepath) as f:
        punch_profile_dict = json.load(f)
    result = {}
    for key, value in punch_profile_dict.items():
        result[int(key)] = PunchProfile(**value)
    return result  # type: ignore

def main() -> None:
    # 定义 main 函数，无返回值
    # ...（略）

def is_opponents_turn() -> bool:
    # 定义 is_opponents_turn 函数，返回布尔值
    return random.randint(1, 10) > 5

def play_round(round_number: int, player: Player, opponent: Player) -> None:
    # 定义 play_round 函数，接收轮数、玩家和对手参数，无返回值
    # ...（略）

if __name__ == "__main__":
    main()
    # 如果当前脚本被直接执行，则调用 main 函数

```