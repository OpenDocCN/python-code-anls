# `basic-computer-games\27_Civil_War\python\Civilwar.py`

```py
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
    # 计算玩家的逃兵数量，逃兵数量等于士气值的倒数
    player.desertions = 100 / player.morale

    # 计算总损失，包括伤亡和逃兵
    loss = player.casualties + player.desertions
    
    # 如果不使用因子，则将当前可用人数设为玩家的可用人数
    if not use_factor:
        present_men: float = player.available_men
    # 否则，调用玩家的获取当前人数方法
    else:
        present_men = player.get_present_men()
    
    # 如果总损失大于等于当前可用人数
    if loss >= present_men:
        # 获取军队因子
        factor = player.get_army_factor()
        # 如果不使用因子，则将因子设为1
        if not use_factor:
            factor = 1
        # 计算伤亡人数，向下取整
        player.casualties = math.floor(13 * player.army_m / 20 * factor)
        # 计算逃兵数量
        player.desertions = 7 * player.casualties / 13
        # 标记为过多损失
        player.excessive_losses = True
# 从用户输入中获取选择，直到用户输入的选择在给定的选项中
def get_choice(prompt: str, choices: List[str]) -> str:
    while True:
        choice = input(prompt)
        if choice in choices:
            break
    return choice


# 计算士气值，返回值越高表示士气越高
def get_morale(stat: PlayerStat, enemy: PlayerStat) -> float:
    """Higher is better"""
    # 计算敌方实力
    enemy_strength = 5 * enemy.army_m / 6
    # 计算士气值
    return (2 * math.pow(stat.food, 2) + math.pow(stat.salaries, 2)) / math.pow(
        enemy_strength, 2
    ) + 1


# 主函数
def main() -> None:
    # 历史数据列表
    historical_data: List[Tuple[str, float, float, float, int, AttackState]] = [
        ("", 0, 0, 0, 0, AttackState.DEFENSIVE),
        ("BULL RUN", 18000, 18500, 1967, 2708, AttackState.DEFENSIVE),
        ("SHILOH", 40000.0, 44894.0, 10699, 13047, AttackState.OFFENSIVE),
        # ... 其他历史数据 ...
        ("ATLANTA", 65000.0, 100000.0, 8500, 3700, AttackState.DEFENSIVE),
    ]
    # 南方策略概率分布
    confederate_strategy_prob_distribution = {}

    # 你会在什么上花钱？
    stats: Dict[int, PlayerStat] = {
        CONF: PlayerStat(),
        UNION: PlayerStat(),
    }

    # 打印标题
    print(" " * 26 + "CIVIL WAR")
    # 打印标题
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

    # 设置南方联盟策略概率分布
    confederate_strategy_prob_distribution[1] = 25
    confederate_strategy_prob_distribution[2] = 25
    confederate_strategy_prob_distribution[3] = 25
    confederate_strategy_prob_distribution[4] = 25
    # 打印空行
    print()
    # 获取用户是否需要游戏说明
    show_instructions = get_choice(
        "DO YOU WANT INSTRUCTIONS? YES OR NO -- ", ["YES", "NO"]
    )

    # 如果用户需要游戏说明
    if show_instructions == "YES":
        # 打印游戏说明
        print()
        print()
        print()
        print()
        print("THIS IS A CIVIL WAR SIMULATION.")
        print("TO PLAY TYPE A RESPONSE WHEN THE COMPUTER ASKS.")
        print("REMEMBER THAT ALL FACTORS ARE INTERRELATED AND THAT YOUR")
        print("RESPONSES COULD CHANGE HISTORY. FACTS AND FIGURES USED ARE")
        print("BASED ON THE ACTUAL OCCURRENCE. MOST BATTLES TEND TO RESULT")
        print("AS THEY DID IN THE CIVIL WAR, BUT IT ALL DEPENDS ON YOU!!")
        print()
        print("THE OBJECT OF THE GAME IS TO WIN AS MANY BATTLES AS ")
        print("POSSIBLE.")
        print()
        print("YOUR CHOICES FOR DEFENSIVE STRATEGY ARE:")
        print("        (1) ARTILLERY ATTACK")
        print("        (2) FORTIFICATION AGAINST FRONTAL ATTACK")
        print("        (3) FORTIFICATION AGAINST FLANKING MANEUVERS")
        print("        (4) FALLING BACK")
        print(" YOUR CHOICES FOR OFFENSIVE STRATEGY ARE:")
        print("        (1) ARTILLERY ATTACK")
        print("        (2) FRONTAL ATTACK")
        print("        (3) FLANKING MANEUVERS")
        print("        (4) ENCIRCLEMENT")
        print("YOU MAY SURRENDER BY TYPING A '5' FOR YOUR STRATEGY."

    # 打印空行
    print()
    print()
    print()
    # 获取是否有两位将军在场
    two_generals = get_choice("(ANSWER YES OR NO) ", ["YES", "NO"]) == "YES"
    # 设置玩家将军的状态为True
    stats[CONF].is_player = True
    # 如果有两位将军
    if two_generals:
        # 确定游戏中的玩家数量
        party: Literal[1, 2] = 2
        # 设置UNION的is_player属性为True
        stats[UNION].is_player = True
    else:
        # 如果只有一位将军，设置party为1
        party = 1
        # 打印提示信息
        print()
        print("YOU ARE THE CONFEDERACY.   GOOD LUCK!")
        print()

    # 打印选择战斗的提示信息
    print("SELECT A BATTLE BY TYPING A NUMBER FROM 1 TO 14 ON")
    print("REQUEST.  TYPE ANY OTHER NUMBER TO END THE SIMULATION.")
    print("BUT '0' BRINGS BACK EXACT PREVIOUS BATTLE SITUATION")
    print("ALLOWING YOU TO REPLAY IT")
    print()
    print("NOTE: A NEGATIVE FOOD$ ENTRY CAUSES THE PROGRAM TO ")
    print("USE THE ENTRIES FROM THE PREVIOUS BATTLE")
    print()
    # 打印提示信息，询问是否需要战斗描述
    print("AFTER REQUESTING A BATTLE, DO YOU WISH ", end="")
    print("BATTLE DESCRIPTIONS ", end="")
    # 获取用户选择
    xs = get_choice("(ANSWER YES OR NO) ", ["YES", "NO"])
    # 初始化变量
    confederacy_lost = 0
    confederacy_win = 0
    # 重置统计数据
    for i in [CONF, UNION]:
        stats[i].p = 0
        stats[i].m = 0
        stats[i].t = 0
        stats[i].available_money = 0
        stats[i].food = 0
        stats[i].salaries = 0
        stats[i].ammunition = 0
        stats[i].strategy = 0
        stats[i].excessive_losses = False
    # 初始化变量
    confederacy_unresolved = 0
    random_nb: float = 0
    # 打印空行
    print()
    print()
    print()
    print()
    print()
    print()
    # 打印战斗结果
    print(
        f"THE CONFEDERACY HAS WON {confederacy_win} BATTLES AND LOST {confederacy_lost}"
    )
    # 判断战争结果
    if stats[CONF].strategy == 5 or (
        stats[UNION].strategy != 5 and confederacy_win <= confederacy_lost
    ):
        print("THE UNION HAS WON THE WAR")
    else:
        print("THE CONFEDERACY HAS WON THE WAR")
    print()
    # 如果南方联盟的胜利次数大于0
    if stats[CONF].r > 0:
        # 打印战斗次数（不包括重新运行的）
        print(
            f"FOR THE {confederacy_win + confederacy_lost + confederacy_unresolved} BATTLES FOUGHT (EXCLUDING RERUNS)"
        )
        # 打印表头
        print(" \t \t ")
        print("CONFEDERACY\t UNION")
        # 打印历史损失
        print(
            f"HISTORICAL LOSSES\t{math.floor(stats[CONF].p + 0.5)}\t{math.floor(stats[UNION].p + 0.5)}"
        )
        # 打印模拟损失
        print(
            f"SIMULATED LOSSES\t{math.floor(stats[CONF].t + 0.5)}\t{math.floor(stats[UNION].t + 0.5)}"
        )
        print()
        # 打印损失占原始值的百分比
        print(
            f"    % OF ORIGINAL\t{math.floor(100 * (stats[CONF].t / stats[CONF].p) + 0.5)}\t{math.floor(100 * (stats[UNION].t / stats[UNION].p) + 0.5)}"
        )
        # 如果不是两位将军
        if not two_generals:
            # 打印南方使用的策略百分比
            print()
            print("UNION INTELLIGENCE SUGGEST THAT THE SOUTH USED")
            print("STRATEGIES 1, 2, 3, 4 IN THE FOLLOWING PERCENTAGES")
            print(
                f"{confederate_strategy_prob_distribution[CONF]} {confederate_strategy_prob_distribution[UNION]} {confederate_strategy_prob_distribution[3]} {confederate_strategy_prob_distribution[4]}"
            )
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```