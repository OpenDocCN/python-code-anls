# `basic-computer-games\15_Boxing\python\boxing.py`

```
    # 设置可执行文件的路径
    #!/usr/bin/env python3
    # 导入 json 模块
    import json
    # 导入 random 模块
    import random
    # 从 dataclasses 模块中导入 dataclass 装饰器
    from dataclasses import dataclass
    # 从 pathlib 模块中导入 Path 类
    from pathlib import Path
    # 从 typing 模块中导入 Dict, Literal, NamedTuple, Tuple 类型
    from typing import Dict, Literal, NamedTuple, Tuple

    # 定义 PunchProfile 类，继承自 NamedTuple
    class PunchProfile(NamedTuple):
        # 定义属性 choices, threshold, hit_damage, block_damage, pre_msg, hit_msg, blocked_msg
        choices: int
        threshold: int
        hit_damage: int
        block_damage: int
        pre_msg: str
        hit_msg: str
        blocked_msg: str
        knockout_possible: bool = False

        # 定义 is_hit 方法，返回布尔值
        def is_hit(self) -> bool:
            return random.randint(1, self.choices) <= self.threshold

    # 定义 Player 类，使用 dataclass 装饰器
    @dataclass
    class Player:
        # 定义属性 name, best, weakness, is_computer, punch_profiles, damage, score, knockedout
        name: str
        best: int  # this hit guarantees 2 damage on opponent
        weakness: int  # you're always hit when your opponent uses this punch
        is_computer: bool
        punch_profiles: Dict[Literal[1, 2, 3, 4], PunchProfile]
        damage: int = 0
        score: int = 0
        knockedout: bool = False

        # 定义 get_punch_choice 方法，返回 Literal[1, 2, 3, 4] 类型
        def get_punch_choice(self) -> Literal[1, 2, 3, 4]:
            if self.is_computer:
                return random.randint(1, 4)  # type: ignore
            else:
                punch = -1
                while punch not in [1, 2, 3, 4]:
                    print(f"{self.name}'S PUNCH", end="? ")
                    punch = int(input())
                return punch  # type: ignore

    # 定义 KNOCKOUT_THRESHOLD 常量
    KNOCKOUT_THRESHOLD = 35

    # 定义 QUESTION_PROMPT 常量
    QUESTION_PROMPT = "? "
    # 定义 KNOCKED_COLD 常量
    KNOCKED_COLD = "{loser} IS KNOCKED COLD AND {winner} IS THE WINNER AND CHAMP"

    # 定义 get_vulnerability 函数，返回整数
    def get_vulnerability() -> int:
        print("WHAT IS HIS VULNERABILITY", end=QUESTION_PROMPT)
        vulnerability = int(input())
        return vulnerability

    # 定义 get_opponent_stats 函数，返回元组
    def get_opponent_stats() -> Tuple[int, int]:
        opponent_best = 0
        opponent_weakness = 0
        while opponent_best == opponent_weakness:
            opponent_best = random.randint(1, 4)
            opponent_weakness = random.randint(1, 4)
        return opponent_best, opponent_weakness

    # 定义 read_punch_profiles 函数，参数为 filepath，返回字典
    def read_punch_profiles(filepath: Path) -> Dict[Literal[1, 2, 3, 4], PunchProfile]:
        # 打开文件并加载 JSON 数据
        with open(filepath) as f:
            punch_profile_dict = json.load(f)
        # 初始化结果字典
        result = {}
    # 遍历 punch_profile_dict 中的键值对
    for key, value in punch_profile_dict.items():
        # 将键转换为整数，并使用值创建 PunchProfile 对象，然后添加到结果字典中
        result[int(key)] = PunchProfile(**value)
    # 返回结果字典
    return result  # type: ignore
# 定义主函数，没有返回值
def main() -> None:
    # 打印标题
    print("BOXING")
    # 打印创意计算的地点
    print("CREATIVE COMPUTING   MORRISTOWN, NEW JERSEY")
    # 打印空行
    print("\n\n")
    # 打印拳击比赛的标题
    print("BOXING OLYMPIC STYLE (3 ROUNDS -- 2 OUT OF 3 WINS)")

    # 获取对手的名字
    print("WHAT IS YOUR OPPONENT'S NAME", end=QUESTION_PROMPT)
    opponent_name = input()
    # 获取玩家的名字
    print("WHAT IS YOUR MAN'S NAME", end=QUESTION_PROMPT)
    player_name = input()

    # 打印不同拳击方式的提示
    print("DIFFERENT PUNCHES ARE 1 FULL SWING 2 HOOK 3 UPPERCUT 4 JAB")
    # 获取玩家最擅长的拳击方式
    print("WHAT IS YOUR MAN'S BEST", end=QUESTION_PROMPT)
    player_best = int(input())  # noqa: TODO - this likely is a bug!
    # 获取玩家的弱点
    player_weakness = get_vulnerability()
    # 创建玩家对象
    player = Player(
        name=player_name,
        best=player_best,
        weakness=player_weakness,
        is_computer=False,
        punch_profiles=read_punch_profiles(
            Path(__file__).parent / "player-profile.json"
        ),
    )

    # 获取对手的最擅长和弱点
    opponent_best, opponent_weakness = get_opponent_stats()
    # 创建对手对象
    opponent = Player(
        name=opponent_name,
        best=opponent_best,
        weakness=opponent_weakness,
        is_computer=True,
        punch_profiles=read_punch_profiles(
            Path(__file__).parent / "opponent-profile.json"
        ),
    )

    # 打印对手的优势和弱点
    print(
        f"{opponent.name}'S ADVANTAGE is {opponent.weakness} AND VULNERABILITY IS SECRET."
    )

    # 进行三轮比赛
    for round_number in (1, 2, 3):
        play_round(round_number, player, opponent)

    # 判断比赛结果并打印
    if player.knockedout:
        print(KNOCKED_COLD.format(loser=player.name, winner=opponent.name))
    elif opponent.knockedout:
        print(KNOCKED_COLD.format(loser=opponent.name, winner=player.name))
    elif opponent.score > player.score:
        print(f"{opponent.name} WINS (NICE GOING), {player.name}")
    else:
        print(f"{player.name} AMAZINGLY WINS")

    # 打印结束语
    print("\n\nAND NOW GOODBYE FROM THE OLYMPIC ARENA.")


# 判断是否轮到对手出拳
def is_opponents_turn() -> bool:
    return random.randint(1, 10) > 5


# 进行每一轮比赛
def play_round(round_number: int, player: Player, opponent: Player) -> None:
    # 打印每一轮比赛的开始
    print(f"ROUND {round_number} BEGINS...\n")
    # 如果对手得分大于等于2或者玩家得分大于等于2，则直接返回，不再执行后续代码
    if opponent.score >= 2 or player.score >= 2:
        return

    # 循环7次，表示进行7个回合的比赛
    for _action in range(7):
        # 如果轮到对手出拳
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

        # 如果出拳是当前玩家的最佳出拳
        if punch == active.best:
            # 被动方受到2点伤害
            passive.damage += 2

        # 打印出拳前的信息
        print(punch_profile.pre_msg.format(active=active, passive=passive), end=" ")
        # 如果被动方的弱点是当前出拳，或者出拳命中
        if passive.weakness == punch or punch_profile.is_hit():
            # 打印出拳命中的信息
            print(punch_profile.hit_msg.format(active=active, passive=passive))
            # 如果可以进行击倒，并且被动方受到的伤害大于击倒阈值
            if punch_profile.knockout_possible and passive.damage > KNOCKOUT_THRESHOLD:
                # 被动方被击倒，跳出循环
                passive.knockedout = True
                break
            # 被动方受到出拳命中的伤害
            passive.damage += punch_profile.hit_damage
        else:
            # 打印出拳被阻挡的信息
            print(punch_profile.blocked_msg.format(active=active, passive=passive))
            # 主动方受到出拳被阻挡的伤害
            active.damage += punch_profile.block_damage

    # 如果玩家或对手被击倒，则直接返回
    if player.knockedout or opponent.knockedout:
        return
    # 如果玩家受到的伤害大于对手受到的伤害
    elif player.damage > opponent.damage:
        # 打印对手获胜的信息，并增加对手的得分
        print(f"{opponent.name} WINS ROUND {round_number}")
        opponent.score += 1
    else:
        # 打印玩家获胜的信息，并增加玩家的得分
        print(f"{player.name} WINS ROUND {round_number}")
        player.score += 1
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```