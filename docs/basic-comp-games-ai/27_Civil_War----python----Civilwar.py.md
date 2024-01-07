# `basic-computer-games\27_Civil_War\python\Civilwar.py`

```

"""
Original game design: Cram, Goodie, Hibbard Lexington H.S.
Modifications: G. Paul, R. Hess (Ties), 1973
"""
# 导入所需的模块
import enum
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

# 定义攻击状态的枚举类型
class AttackState(enum.Enum):
    DEFENSIVE = 1
    BOTH_OFFENSIVE = 2
    OFFENSIVE = 3

# 定义常量
CONF = 1
UNION = 2

# 定义玩家状态的数据类
@dataclass
class PlayerStat:
    food: float = 0
    salaries: float = 0
    ammunition: float = 0
    desertions: float = 0
    casualties: float = 0
    morale: float = 0
    strategy: int = 0
    available_men: int = 0
    available_money: int = 0
    army_c: float = 0
    army_m: float = 0  # available_men ????
    inflation: float = 0
    r: float = 0
    t: float = 0  # casualties + desertions
    q: float = 0  # accumulated cost?
    p: float = 0
    m: float = 0
    is_player = False
    excessive_losses = False

    # 设置可用资金
    def set_available_money(self):
        if self.is_player:
            factor = 1 + (self.r - self.q) / (self.r + 1)
        else:
            factor = 1
        self.available_money = 100 * math.floor(
            (self.army_m * (100 - self.inflation) / 2000) * factor + 0.5
        )

    # 获取成本
    def get_cost(self) -> float:
        return self.food + self.salaries + self.ammunition

    # 获取军队因素
    def get_army_factor(self) -> float:
        return 1 + (self.p - self.t) / (self.m + 1)

    # 获取当前人数
    def get_present_men(self) -> float:
        return self.army_m * self.get_army_factor()

# 模拟玩家损失
def simulate_losses(player1: PlayerStat, player2: PlayerStat) -> float:
    """Simulate losses of player 1"""
    tmp = (2 * player1.army_c / 5) * (
        1 + 1 / (2 * (abs(player1.strategy - player2.strategy) + 1))
    )
    tmp = tmp * (1.28 + (5 * player1.army_m / 6) / (player1.ammunition + 1))
    tmp = math.floor(tmp * (1 + 1 / player1.morale) + 0.5)
    return tmp

# 更新军队
def update_army(player: PlayerStat, enemy: PlayerStat, use_factor=False) -> None:
    player.casualties = simulate_losses(player, enemy)
    player.desertions = 100 / player.morale
    loss = player.casualties + player.desertions
    if not use_factor:
        present_men: float = player.available_men
    else:
        present_men = player.get_present_men()
    if loss >= present_men:
        factor = player.get_army_factor()
        if not use_factor:
            factor = 1
        player.casualties = math.floor(13 * player.army_m / 20 * factor)
        player.desertions = 7 * player.casualties / 13
        player.excessive_losses = True

# 获取选择
def get_choice(prompt: str, choices: List[str]) -> str:
    while True:
        choice = input(prompt)
        if choice in choices:
            break
    return choice

# 获取士气
def get_morale(stat: PlayerStat, enemy: PlayerStat) -> float:
    """Higher is better"""
    enemy_strength = 5 * enemy.army_m / 6
    return (2 * math.pow(stat.food, 2) + math.pow(stat.salaries, 2)) / math.pow(
        enemy_strength, 2
    ) + 1

# 主函数
if __name__ == "__main__":
    main()

```