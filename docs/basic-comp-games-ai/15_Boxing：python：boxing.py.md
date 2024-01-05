# `15_Boxing\python\boxing.py`

```
#!/usr/bin/env python3  # 指定脚本的解释器为 Python 3

import json  # 导入 json 模块，用于处理 JSON 数据
import random  # 导入 random 模块，用于生成随机数
from dataclasses import dataclass  # 导入 dataclass 装饰器，用于创建数据类
from pathlib import Path  # 导入 Path 类，用于处理文件路径
from typing import Dict, Literal, NamedTuple, Tuple  # 导入类型提示，用于类型检查
    def is_hit(self) -> bool:  # 定义一个方法 is_hit，返回布尔值
        return random.randint(1, self.choices) <= self.threshold  # 返回一个随机数是否小于等于阈值的布尔值


@dataclass  # 使用 dataclass 装饰器
class Player:  # 定义一个类 Player
    name: str  # 玩家的名字
    best: int  # 最佳命中，保证对对手造成2点伤害
    weakness: int  # 弱点，当对手使用这种拳时，你总是会被命中
    is_computer: bool  # 是否是电脑玩家

    # 对于每种拳类型，我们有命中的概率
    punch_profiles: Dict[Literal[1, 2, 3, 4], PunchProfile]  # 拳击概况的字典，键为拳击类型，值为 PunchProfile 类型

    damage: int = 0  # 伤害，默认值为0
    score: int = 0  # 得分，默认值为0
    knockedout: bool = False  # 是否被击倒，默认值为False

    def get_punch_choice(self) -> Literal[1, 2, 3, 4]:  # 定义一个方法 get_punch_choice，返回拳击类型
        if self.is_computer:  # 如果是电脑玩家
# 返回一个1到4之间的随机整数，表示拳击手的出拳动作
def get_punch() -> int:
    return random.randint(1, 4)  # type: ignore

# 如果拳击手不是随机出拳，则根据用户输入获取拳击手的出拳动作
# 如果用户输入不在1到4之间的数字，则要求用户重新输入
def get_punch() -> int:
    else:
        punch = -1
        while punch not in [1, 2, 3, 4]:
            print(f"{self.name}'S PUNCH", end="? ")
            punch = int(input())
        return punch  # type: ignore

# 表示被击倒的阈值
KNOCKOUT_THRESHOLD = 35

# 问题提示符
QUESTION_PROMPT = "? "
# 表示被击倒后的提示信息
KNOCKED_COLD = "{loser} IS KNOCKED COLD AND {winner} IS THE WINNER AND CHAMP"

# 获取拳击手的脆弱性值
def get_vulnerability() -> int:
    print("WHAT IS HIS VULNERABILITY", end=QUESTION_PROMPT)
    vulnerability = int(input())
    return vulnerability
def get_opponent_stats() -> Tuple[int, int]:
    opponent_best = 0  # 初始化对手的最佳属性
    opponent_weakness = 0  # 初始化对手的弱点属性
    while opponent_best == opponent_weakness:  # 当对手的最佳属性和弱点属性相同时
        opponent_best = random.randint(1, 4)  # 随机生成对手的最佳属性
        opponent_weakness = random.randint(1, 4)  # 随机生成对手的弱点属性
    return opponent_best, opponent_weakness  # 返回对手的最佳属性和弱点属性


def read_punch_profiles(filepath: Path) -> Dict[Literal[1, 2, 3, 4], PunchProfile]:
    with open(filepath) as f:  # 打开指定文件
        punch_profile_dict = json.load(f)  # 从文件中加载 PunchProfile 字典
    result = {}  # 初始化结果字典
    for key, value in punch_profile_dict.items():  # 遍历 PunchProfile 字典
        result[int(key)] = PunchProfile(**value)  # 将 PunchProfile 对象添加到结果字典中
    return result  # 返回 PunchProfile 字典


def main() -> None:
    # 打印"BOXING"
    print("BOXING")
    # 打印"CREATIVE COMPUTING   MORRISTOWN, NEW JERSEY"
    print("CREATIVE COMPUTING   MORRISTOWN, NEW JERSEY")
    # 打印两个换行符
    print("\n\n")
    # 打印"BOXING OLYMPIC STYLE (3 ROUNDS -- 2 OUT OF 3 WINS)"
    print("BOXING OLYMPIC STYLE (3 ROUNDS -- 2 OUT OF 3 WINS)")

    # 打印"WHAT IS YOUR OPPONENT'S NAME"，并等待用户输入对手的名字
    print("WHAT IS YOUR OPPONENT'S NAME", end=QUESTION_PROMPT)
    opponent_name = input()
    # 打印"WHAT IS YOUR MAN'S NAME"，并等待用户输入自己的名字
    print("WHAT IS YOUR MAN'S NAME", end=QUESTION_PROMPT)
    player_name = input()

    # 打印"DIFFERENT PUNCHES ARE 1 FULL SWING 2 HOOK 3 UPPERCUT 4 JAB"
    print("DIFFERENT PUNCHES ARE 1 FULL SWING 2 HOOK 3 UPPERCUT 4 JAB")
    # 打印"WHAT IS YOUR MAN'S BEST"，并等待用户输入自己最擅长的拳法
    print("WHAT IS YOUR MAN'S BEST", end=QUESTION_PROMPT)
    player_best = int(input())  # noqa: TODO - this likely is a bug!
    # 获取玩家的弱点
    player_weakness = get_vulnerability()
    # 创建玩家对象，包括名字、最擅长的拳法、弱点等信息
    player = Player(
        name=player_name,
        best=player_best,
        weakness=player_weakness,
        is_computer=False,
        punch_profiles=read_punch_profiles(
    # 读取当前文件所在目录下的 player-profile.json 文件的内容
    Path(__file__).parent / "player-profile.json"
),

opponent_best, opponent_weakness = get_opponent_stats()
# 创建对手对象，包括对手的名字、优势和劣势，以及是否是电脑
opponent = Player(
    name=opponent_name,
    best=opponent_best,
    weakness=opponent_weakness,
    is_computer=True,
    # 读取当前文件所在目录下的 opponent-profile.json 文件的内容作为对手的拳击数据
    punch_profiles=read_punch_profiles(
        Path(__file__).parent / "opponent-profile.json"
    ),
)

# 打印对手的名字、劣势，并且暴露对手的弱点是秘密
print(
    f"{opponent.name}'S ADVANTAGE is {opponent.weakness} AND VULNERABILITY IS SECRET."
)

# 循环进行三轮比赛
for round_number in (1, 2, 3):
        play_round(round_number, player, opponent)  # 调用play_round函数，传入轮数、玩家和对手作为参数

    if player.knockedout:  # 如果玩家被击倒
        print(KNOCKED_COLD.format(loser=player.name, winner=opponent.name))  # 打印玩家被击倒的消息
    elif opponent.knockedout:  # 如果对手被击倒
        print(KNOCKED_COLD.format(loser=opponent.name, winner=player.name))  # 打印对手被击倒的消息
    elif opponent.score > player.score:  # 如果对手得分高于玩家得分
        print(f"{opponent.name} WINS (NICE GOING), {player.name}")  # 打印对手获胜的消息
    else:  # 其他情况
        print(f"{player.name} AMAZINGLY WINS")  # 打印玩家获胜的消息

    print("\n\nAND NOW GOODBYE FROM THE OLYMPIC ARENA.")  # 打印结束比赛的消息


def is_opponents_turn() -> bool:  # 定义函数is_opponents_turn，返回布尔值
    return random.randint(1, 10) > 5  # 返回随机数是否大于5的布尔值


def play_round(round_number: int, player: Player, opponent: Player) -> None:  # 定义函数play_round，接受轮数、玩家和对手作为参数，返回空值
    print(f"ROUND {round_number} BEGINS...\n")  # 打印当前轮数开始的消息
    # 如果对手得分大于等于2或者玩家得分大于等于2，则直接返回，不执行后续代码
    if opponent.score >= 2 or player.score >= 2:
        return

    # 循环7次，表示每个动作的可能性
    for _action in range(7):
        # 如果轮到对手行动
        if is_opponents_turn():
            # 获取对手的出拳选择
            punch = opponent.get_punch_choice()
            # 设置主动方为对手，被动方为玩家
            active = opponent
            passive = player
        else:
            # 获取玩家的出拳选择
            punch = player.get_punch_choice()
            # 设置主动方为玩家，被动方为对手
            active = player
            passive = opponent

        # 加载当前玩家出拳的打击特征
        punch_profile = active.punch_profiles[punch]

        # 如果出拳选择为主动方的最佳出拳
        if punch == active.best:
            # 增加被动方的伤害值
            passive.damage += 2

        # 打印出拳特征的前置信息，使用active和passive进行格式化，不换行
        print(punch_profile.pre_msg.format(active=active, passive=passive), end=" ")
        # 如果被动角色的弱点是拳击或者拳击配置命中
        if passive.weakness == punch or punch_profile.is_hit():
            # 打印拳击命中信息
            print(punch_profile.hit_msg.format(active=active, passive=passive))
            # 如果可能被击倒并且受到的伤害大于击倒阈值
            if punch_profile.knockout_possible and passive.damage > KNOCKOUT_THRESHOLD:
                # 被动角色被击倒
                passive.knockedout = True
                # 跳出循环
                break
            # 被动角色受到拳击伤害
            passive.damage += punch_profile.hit_damage
        else:
            # 打印拳击被阻挡信息
            print(punch_profile.blocked_msg.format(active=active, passive=passive))
            # 主动角色受到阻挡伤害
            active.damage += punch_profile.block_damage

    # 如果玩家或对手被击倒，则返回
    if player.knockedout or opponent.knockedout:
        return
    # 如果玩家受到的伤害大于对手受到的伤害
    elif player.damage > opponent.damage:
        # 打印对手获胜信息
        print(f"{opponent.name} WINS ROUND {round_number}")
        # 对手得分加一
        opponent.score += 1
    else:
        # 打印玩家获胜信息
        print(f"{player.name} WINS ROUND {round_number}")
        # 玩家得分加一
        player.score += 1
if __name__ == "__main__":
    # 如果当前脚本被直接执行，而不是被导入到其他模块中，则执行下面的代码
    main()  # 调用名为main的函数
```