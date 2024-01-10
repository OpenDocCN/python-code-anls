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

NB_PLAYERS = 6  # 定义常量 NB_PLAYERS 为 6，表示球队中的球员数量


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
    name: str  # 球队名称
    players: List[str]  # 6 个球员
    shots_on_net: int = 0  # 射门次数，默认为 0
    goals: List[int] = field(default_factory=lambda: [0 for _ in range(NB_PLAYERS)])  # 进球数，默认为 0
    assists: List[int] = field(default_factory=lambda: [0 for _ in range(NB_PLAYERS)])  # 助攻数，默认为 0
    score: int = 0  # 得分，默认为 0

    def show_lineup(self) -> None:  # 定义 show_lineup 方法，无返回值
        print(" " * 10 + f"{self.name} STARTING LINEUP")  # 打印球队名称和 STARTING LINEUP
        for player in self.players:  # 遍历球队中的球员
            print(player)  # 打印球员名称


def ask_binary(prompt: str, error_msg: str) -> bool:  # 定义 ask_binary 函数，接收两个字符串参数，返回布尔值
    while True:  # 进入循环
        answer = input(prompt).lower()  # 获取用户输入并转换为小写
        if answer in ["y", "yes"]:  # 如果用户输入为 "y" 或 "yes"
            return True  # 返回 True
        if answer in ["n", "no"]:  # 如果用户输入为 "n" 或 "no"
            return False  # 返回 False
        print(error_msg)  # 打印错误消息


def get_team_names() -> Tuple[str, str]:  # 定义 get_team_names 函数，返回元组类型
    while True:  # 进入循环
        answer = input("ENTER THE TWO TEAMS: ")  # 获取用户输入
        if answer.count(",") == 1:  # 如果输入中包含一个逗号
            return answer.split(",")  # 返回逗号分隔的两个字符串作为元组
        print("separated by a single comma")  # 打印提示信息


def get_pass() -> int:  # 定义 get_pass 函数，返回整数类型
    while True:  # 进入循环
        answer = input("PASS? ")  # 获取用户输入
        try:  # 尝试转换用户输入为整数
            passes = int(answer)  # 转换为整数
            if passes >= 0 and passes <= 3:  # 如果输入在 0 到 3 之间
                return passes  # 返回输入的整数
        except ValueError:  # 捕获值错误异常
            print("ENTER A NUMBER BETWEEN 0 AND 3")  # 打印错误提示信息
# 获取每场比赛的分钟数
def get_minutes_per_game() -> int:
    # 无限循环，直到输入有效的分钟数
    while True:
        # 获取用户输入的分钟数
        answer = input("ENTER THE NUMBER OF MINUTES IN A GAME ")
        try:
            # 尝试将输入转换为整数
            minutes = int(answer)
            # 如果分钟数大于等于1，则返回该分钟数
            if minutes >= 1:
                return minutes
        # 如果输入的不是整数，则捕获 ValueError 异常
        except ValueError:
            print("ENTER A NUMBER")


# 获取球员姓名列表
def get_player_names(prompt: str) -> List[str]:
    players = []
    # 打印提示信息
    print(prompt)
    # 循环获取球员姓名，最多获取6个
    for i in range(1, 7):
        player = input(f"PLAYER {i}: ")
        players.append(player)
    return players


# 进行投篮动作
def make_shot(
    controlling_team: int, team_a: Team, team_b: Team, player_index: List[int], j: int
) -> Tuple[int, int, int, int]:
    # 无限循环，直到输入有效的投篮动作
    while True:
        try:
            s = int(input("SHOT? "))
        except ValueError:
            continue
        # 如果投篮动作在1到4之间，则跳出循环
        if s >= 1 and s <= 4:
            break
    # 根据控制球队打印球员姓名
    if controlling_team == 1:
        print(team_a.players[player_index[j - 1]])
    else:
        print(team_b.players[player_index[j - 1]])
    g = player_index[j - 1]
    g1 = 0
    g2 = 0
    # 根据投篮动作选择不同的逻辑
    if s == 1:
        print(" LET'S A BOOMER GO FROM THE RED LINE!!\n")  # line 400
        z = 10
    elif s == 2:
        print(" FLIPS A WRISTSHOT DOWN THE ICE\n")  # line 420
        # 可能缺失了原始代码中的第430行
    elif s == 3:
        print(" BACKHANDS ONE IN ON THE GOALTENDER\n")
        z = 25
    elif s == 4:
        print(" SNAPS A LONG FLIP SHOT\n")
        # line 460
        z = 17
    return z, g, g1, g2


# 打印标题
def print_header() -> None:
    print(" " * 33 + "HOCKEY")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")


# 打印游戏说明
def instructions() -> None:
    # 调用 ask_binary 函数询问用户是否需要游戏说明
    wants_it = ask_binary("WOULD YOU LIKE THE INSTRUCTIONS? ", "ANSWER YES OR NO!!")
    # 如果用户想要进行模拟曲棍球比赛
    if wants_it:
        # 打印空行
        print()
        # 打印提示信息：这是一个模拟的曲棍球比赛
        print("THIS IS A SIMULATED HOCKEY GAME.")
        # 打印提示信息：问题和回答
        print("QUESTION     RESPONSE")
        # 打印提示信息：传球，输入0到3之间的数字表示要传球的次数
        print("PASS         TYPE IN THE NUMBER OF PASSES YOU WOULD")
        print("             LIKE TO MAKE, FROM 0 TO 3.")
        # 打印提示信息：射门，输入1到4之间的数字表示要进行的射门类型
        print("SHOT         TYPE THE NUMBER CORRESPONDING TO THE SHOT")
        print("             YOU WANT TO MAKE.  ENTER:")
        print("             1 FOR A SLAPSHOT")
        print("             2 FOR A WRISTSHOT")
        print("             3 FOR A BACKHAND")
        print("             4 FOR A SNAP SHOT")
        # 打印提示信息：射门区域，输入1到4之间的数字表示瞄准的射门区域
        print("AREA         TYPE IN THE NUMBER CORRESPONDING TO")
        print("             THE AREA YOU ARE AIMING AT.  ENTER:")
        print("             1 FOR UPPER LEFT HAND CORNER")
        print("             2 FOR UPPER RIGHT HAND CORNER")
        print("             3 FOR LOWER LEFT HAND CORNER")
        print("             4 FOR LOWER RIGHT HAND CORNER")
        # 打印提示信息：在比赛开始时，需要输入球员的名字
        print("AT THE START OF THE GAME, YOU WILL BE ASKED FOR THE NAMES")
        print("OF YOUR PLAYERS.  THEY ARE ENTERED IN THE ORDER: ")
        print("LEFT WING, CENTER, RIGHT WING, LEFT DEFENSE,")
        print("RIGHT DEFENSE, GOALKEEPER.  ANY OTHER INPUT REQUIRED WILL")
        print("HAVE EXPLANATORY INSTRUCTIONS.")
# 定义一个名为 team1_action 的函数，接受 pass_value, player_index, team_a, team_b, j 作为参数，并返回一个元组
def team1_action(
    pass_value: int, player_index: List[int], team_a: Team, team_b: Team, j: int
) -> Tuple[int, int, int, int]:
    # 如果传递值为1
    if pass_value == 1:
        # 打印球员传球情况
        print(
            team_a.players[player_index[j - 2]]
            + " LEADS "
            + team_a.players[player_index[j - 1]]
            + " WITH A PERFECT PASS.\n"
        )
        # 打印球员突破情况
        print(team_a.players[player_index[j - 1]] + " CUTTING IN!!!\n")
        # 设置得分球员、助攻球员1、助攻球员2和 z1 的值
        scoring_player = player_index[j - 1]
        goal_assistant1 = player_index[j - 2]
        goal_assistant2 = 0
        z1 = 3
    # 如果传递值为2
    elif pass_value == 2:
        # 打印球员传球情况
        print(
            team_a.players[player_index[j - 2]]
            + " GIVES TO A STREAKING "
            + team_a.players[player_index[j - 1]]
        )
        # 打印球员突破情况
        print(
            team_a.players[player_index[j - 3]]
            + " COMES DOWN ON "
            + team_b.players[4]
            + " AND "
            + team_b.players[3]
        )
        # 设置得分球员、助攻球员1、助攻球员2和 z1 的值
        scoring_player = player_index[j - 3]
        goal_assistant1 = player_index[j - 1]
        goal_assistant2 = player_index[j - 2]
        z1 = 2
    # 如果传递值为3
    elif pass_value == 3:
        # 打印球员传球情况
        print("OH MY GOD!! A ' 4 ON 2 ' SITUATION\n")
        print(
            team_a.players[player_index[j - 3]]
            + " LEADS "
            + team_a.players[player_index[j - 2]]
            + "\n"
        )
        # 打印球员突破情况
        print(team_a.players[player_index[j - 2]] + " IS WHEELING THROUGH CENTER.\n")
        print(
            team_a.players[player_index[j - 2]]
            + " GIVES AND GOEST WITH "
            + team_a.players[player_index[j - 1]]
        )
        print("PRETTY PASSING!\n")
        print(
            team_a.players[player_index[j - 1]]
            + " DROPS IT TO "
            + team_a.players[player_index[j - 4]]
        )
        # 设置得分球员、助攻球员1、助攻球员2和 z1 的值
        scoring_player = player_index[j - 4]
        goal_assistant1 = player_index[j - 1]
        goal_assistant2 = player_index[j - 2]
        z1 = 1
    # 返回得分球员、助攻球员1、助攻球员2和 z1 的值
    return scoring_player, goal_assistant1, goal_assistant2, z1


def team2_action(
    # 定义函数参数，pass_value为整数类型，player_index为整数列表类型，team_a和team_b为Team类型，j为整数类型
def final_message(team_a: Team, team_b: Team, player_index: List[int]) -> None:
    # 输出比赛结束的提示音
    print("THAT'S THE SIREN\n")
    # 输出空行
    print("\n")
    # 输出最终比分的标题
    print(" " * 15 + "FINAL SCORE:\n")
    # 如果B队得分小于等于A队得分
    if team_b.score <= team_a.score:
        # 输出A队和B队的最终比分
        print(f"{team_a.name}: {team_a.score}\t{team_b.name}: {team_b.score}\n")
    # 如果不是平局，则打印队伍B的名称和得分，队伍A的名称和得分
    else:
        print(f"{team_b.name}: {team_b.score}\t{team_a.name}\t:{team_a.score}\n")
    # 打印空行
    print("\n")
    # 打印得分总结
    print(" " * 10 + "SCORING SUMMARY\n")
    # 打印空行
    print("\n")
    # 打印队伍A的名称
    print(" " * 25 + team_a.name + "\n")
    # 打印球员的名称、进球数和助攻数
    print("\tNAME\tGOALS\tASSISTS\n")
    # 打印分隔线
    print("\t----\t-----\t-------\n")
    # 遍历队伍A的前5名球员，打印他们的名称、进球数和助攻数
    for i in range(1, 6):
        print(f"\t{team_a.players[i]}\t{team_a.goals[i]}\t{team_a.assists[i]}\n")
    # 打印空行
    print("\n")
    # 打印队伍B的名称
    print(" " * 25 + team_b.name + "\n")
    # 打印球员的名称、进球数和助攻数
    print("\tNAME\tGOALS\tASSISTS\n")
    # 打印分隔线
    print("\t----\t-----\t-------\n")
    # 遍历队伍B的前5名球员，打印他们的名称、进球数和助攻数
    for t in range(1, 6):
        print(f"\t{team_b.players[t]}\t{team_b.goals[t]}\t{team_b.assists[t]}\n")
    # 打印空行
    print("\n")
    # 打印射门次数
    print("SHOTS ON NET\n")
    # 打印队伍A的射门次数
    print(f"{team_a.name}: {team_a.shots_on_net}\n")
    # 打印队伍B的射门次数
    print(team_b.name + f": {team_b.shots_on_net}\n")
def main() -> None:
    # 打印游戏介绍
    print_header()
    # 创建一个长度为21的整数列表，初始化为0
    player_index: List[int] = [0 for _ in range(21)]
    # 打印三行空行
    print("\n" * 3)
    # 打印游戏说明
    instructions()

    # 收集输入信息
    # 获取两支队伍的名称
    team_name_a, team_name_b = get_team_names()
    print()
    # 获取每场比赛的分钟数
    minutes_per_game = get_minutes_per_game()
    print()
    # 获取A队球员名单
    players_a = get_player_names(f"WOULD THE {team_name_a} COACH ENTER HIS TEAM")
    print()
    # 获取B队球员名单
    players_b = get_player_names(f"WOULD THE {team_name_b} COACH DO THE SAME")
    # 创建A队和B队的Team对象
    team_a = Team(team_name_a, players_a)
    team_b = Team(team_name_b, players_b)
    print()
    # 输入裁判的名字
    referee = input("INPUT THE REFEREE FOR THIS GAME: ")
    print()
    # 展示A队的阵容
    team_a.show_lineup()
    print()
    # 展示B队的阵容
    team_b.show_lineup()
    # 打印比赛开始的提示
    print("WE'RE READY FOR TONIGHTS OPENING FACE-OFF.")
    # 打印裁判将在哪两名球员之间进行开球
    print(
        f"{referee} WILL DROP THE PUCK BETWEEN "
        f"{team_a.players[0]} AND {team_b.players[0]}"
    )
    # 初始化剩余时间为比赛总时间
    remaining_time = minutes_per_game

    # 进行比赛
    while remaining_time > 0:
        # 模拟一轮比赛，并更新剩余时间
        cont, remaining_time = simulate_game_round(
            team_a, team_b, player_index, remaining_time
        )
        remaining_time -= 1
        # 如果cont为"break"，则跳出循环
        if cont == "break":
            break

    # 打印比赛结束的信息
    final_message(team_a, team_b, player_index)


def handle_hit(
    controlling_team: int,
    team_a: Team,
    team_b: Team,
    player_index: List[int],
    goal_player: int,
    goal_assistant1: int,
    goal_assistant2: int,
    hit_area: int,
    z: int,
) -> int:
    # 处理球员的击打动作
    while True:
        # 生成一个1到100的随机整数，赋值给player_index[20]
        player_index[20] = randint(1, 100)
        # 如果player_index[20]不能被z整除，则跳出循环
        if player_index[20] % z != 0:
            break
        # 生成一个1到100的随机整数，赋值给a2
        a2 = randint(1, 100)
        # 如果a2能被4整除
        if a2 % 4 == 0:
            # 如果控制球队为1，则打印救球信息和REBOUND
            if controlling_team == 1:
                print(f"SAVE {team_b.players[5]} --  REBOUND\n")
            # 否则打印救球信息和FOLLOW up
            else:
                print(f"SAVE {team_a.players[5]} --  FOLLOW up\n")
            # 继续下一轮循环
            continue
        else:
            # 击打区域加1
            hit_area += 1
    # 如果球员索引为20的球员得分除以z的余数不为0
    if player_index[20] % z != 0:
        # 如果控制球队为1
        if controlling_team == 1:
            # 打印球队A进球的消息
            print(f"GOAL {team_a.name}\n")
            # 球队A得分加1
            team_a.score += 1
        else:
            # 打印球队B得分的消息
            print(f"SCORE {team_b.name}\n")
            # 球队B得分加1
            team_b.score += 1
        # 打印空行
        print("\n")
        # 打印"SCORE: "
        print("SCORE: ")
        # 如果球队B得分小于等于球队A得分
        if team_b.score <= team_a.score:
            # 打印球队A和球队B的得分
            print(f"{team_a.name}: {team_a.score}\t{team_b.name}: {team_b.score}\n")
        else:
            # 打印球队B和球队A的得分
            print(f"{team_b.name}: {team_b.score}\t{team_a.name}: {team_a.score}\n")
        # 如果控制球队为1
        if controlling_team == 1:
            # 球队为球队A
            team = team_a
        else:
            # 球队为球队B
            team = team_b
        # 打印进球球员的消息
        print("GOAL SCORED BY: " + team.players[goal_player] + "\n")
        # 如果有第一个助攻者
        if goal_assistant1 != 0:
            # 如果有第二个助攻者
            if goal_assistant2 != 0:
                # 打印两个助攻者的消息
                print(
                    f" ASSISTED BY: {team.players[goal_assistant1]}"
                    f" AND {team.players[goal_assistant2]}"
                )
            else:
                # 打印第一个助攻者的消息
                print(f" ASSISTED BY: {team.players[goal_assistant1]}")
            # 球队助攻者1的助攻数加1
            team.assists[goal_assistant1] += 1
            # 球队助攻者2的助攻数加1
            team.assists[goal_assistant2] += 1
        else:
            # 打印无助攻的消息
            print(" UNASSISTED.\n")
        # 进球球员的进球数加1
        team.goals[goal_player] += 1

    # 返回击球区域
    return hit_area
# 处理球门失误事件，返回一个元组，包含字符串和整数
def handle_miss(
    controlling_team: int,  # 控制球队的编号
    team_a: Team,  # 球队A对象
    team_b: Team,  # 球队B对象
    remaining_time: int,  # 剩余时间
    goal_player: int,  # 进球球员的编号
) -> Tuple[str, int]:  # 返回一个元组，包含字符串和整数
    saving_player = randint(1, 7)  # 随机生成一个1到7的整数，表示扑救球员
    if controlling_team == 1:  # 如果控制球队是1
        if saving_player == 1:  # 如果扑救球员是1
            print("KICK SAVE AND A BEAUTY BY " + team_b.players[5] + "\n")  # 打印扑救球员的动作
            print("CLEARED OUT BY " + team_b.players[3] + "\n")  # 打印清除球员的动作
            remaining_time -= 1  # 剩余时间减1
            return ("continue", remaining_time)  # 返回继续比赛和剩余时间
        if saving_player == 2:  # 如果扑救球员是2
            print("WHAT A SPECTACULAR GLOVE SAVE BY " + team_b.players[5] + "\n")  # 打印扑救球员的动作
            print("AND " + team_b.players[5] + " GOLFS IT INTO THE CROWD\n")  # 打印球员的动作
            return ("break", remaining_time)  # 返回中断比赛和剩余时间
        # 其他情况依此类推，省略注释
    else:
        # 如果救球的球员是1号球员
        if saving_player == 1:
            # 打印提示信息，表示1号球员挡住了球
            print(f"STICK SAVE BY {team_a.players[5]}\n")
            # 打印提示信息，表示1号球员将球清出了危险区域
            print(f"AND CLEARED OUT BY {team_a.players[3]}\n")
            # 剩余时间减1
            remaining_time -= 1
            # 返回继续比赛的状态和剩余时间
            return ("continue", remaining_time)
        # 如果救球的球员是2号球员
        if saving_player == 2:
            # 打印提示信息，表示2号球员将球打在了门柱上
            print(
                "OH MY GOD!! "
                f"{team_b.players[goal_player]} RATTLES ONE OFF THE POST\n"
            )
            # 打印提示信息，表示2号球员将球打到了右边，1号球员将球挡住了
            print(
                f"TO THE RIGHT OF {team_a.players[5]} AND "
                f"{team_a.players[5]} COVERS "
            )
            # 打印提示信息，表示球在场上滚动
            print("ON THE LOOSE PUCK!\n")
            # 返回暂停比赛的状态和剩余时间
            return ("break", remaining_time)
        # 如果救球的球员是3号球员
        if saving_player == 3:
            # 打印提示信息，表示3号球员用溜冰鞋将球挡住
            print("SKATE SAVE BY " + team_a.players[5] + "\n")
            # 打印提示信息，表示3号球员将球打入看台
            print(team_a.players[5] + " WHACKS THE LOOSE PUCK INTO THE STANDS\n")
            # 返回暂停比赛的状态和剩余时间
            return ("break", remaining_time)
        # 如果救球的球员是4号球员
        if saving_player == 4:
            # 打印提示信息，表示4号球员用球棒将球挡住，并将球清出危险区域
            print(
                "STICK SAVE BY " + team_a.players[5] + " AND HE CLEARS IT OUT HIMSELF\n"
            )
            # 剩余时间减1
            remaining_time -= 1
            # 返回继续比赛的状态和剩余时间
            return ("continue", remaining_time)
        # 如果救球的球员是5号球员
        if saving_player == 5:
            # 打印提示信息，表示5号球员用脚将球踢出
            print("KICKED OUT BY " + team_a.players[5] + "\n")
            # 打印提示信息，表示球弹到了中线
            print("AND IT REBOUNDS ALL THE WAY TO CENTER ICE\n")
            # 剩余时间减1
            remaining_time -= 1
            # 返回继续比赛的状态和剩余时间
            return ("continue", remaining_time)
        # 如果救球的球员是6号球员
        if saving_player == 6:
            # 打印提示信息，表示6号球员用手套将球挡住，并抓住了球
            print("GLOVE SAVE " + team_a.players[5] + " AND HE HANGS ON\n")
            # 返回暂停比赛的状态和剩余时间
            return ("break", remaining_time)
    # 返回继续比赛的状态和剩余时间
    return ("continue", remaining_time)
# 模拟一局比赛
def simulate_game_round(
    team_a: Team, team_b: Team, player_index: List[int], remaining_time: int
) -> Tuple[str, int]:
    # 随机确定控球队伍
    controlling_team = randint(1, 2)
    # 根据控球队伍输出信息
    if controlling_team == 1:
        print(f"{team_a.name} HAS CONTROL OF THE PUCK.")
    else:
        print(f"{team_b.name} HAS CONTROL.")
    # 获取传球值
    pass_value = get_pass()
    # 重置球员索引
    for i in range(1, 4):
        player_index[i] = 0

    # 进入循环，直到满足条件跳出
    while True:  # Line 310
        j = 0
        for j in range(1, (pass_value + 2) + 1):
            player_index[j] = randint(1, 5)
        # 检查传球是否合法，如果合法则跳出循环
        if player_index[j - 1] == player_index[j - 2] or (
            pass_value + 2 >= 3
            and (
                player_index[j - 1] == player_index[j - 3]
                or player_index[j - 2] == player_index[j - 3]
            )
        ):
            break
    # 如果传球值为0，进行射门
    if pass_value == 0:  # line 350
        # 进行射门并返回结果
        z, goal_player, goal_assistant1, goal_assistant2 = make_shot(
            controlling_team, team_a, team_b, player_index, j
        )
    else:
        # 如果控制球队为1
        if controlling_team == 1:
            # 调用team1_action函数，获取进球球员、助攻球员1、助攻球员2和z1的值
            goal_player, goal_assistant1, goal_assistant2, z1 = team1_action(
                pass_value, player_index, team_a, team_b, j
            )
        else:
            # 否则调用team2_action函数，获取进球球员、助攻球员1、助攻球员2和z1的值
            goal_player, goal_assistant1, goal_assistant2, z1 = team2_action(
                pass_value, player_index, team_a, team_b, j
            )
        # 循环直到输入合法的射门类型
        while True:
            shot_type = int(input("SHOT? "))
            if not (shot_type < 1 or shot_type > 4):
                break
        # 如果控制球队为1，打印进球球员的名字
        if controlling_team == 1:
            print(team_a.players[goal_player], end="")
        else:
            print(team_b.players[goal_player], end="")
        # 根据射门类型打印相应的信息，并计算z的值
        if shot_type == 1:
            print(" LET'S A BIG SLAP SHOT GO!!\n")
            z = 4
            z += z1
        if shot_type == 2:
            print(" RIPS A WRIST SHOT OFF\n")
            z = 2
            z += z1
        if shot_type == 3:
            print(" GETS A BACKHAND OFF\n")
            z = 3
            z += z1
        if shot_type == 4:
            print(" SNAPS OFF A SNAP SHOT\n")
            z = 2
            z += z1
    # 循环直到输入合法的射门区域
    while True:
        goal_area = int(input("AREA? "))
        if not (goal_area < 1 or goal_area > 4):
            break
    # 根据控制球队更新射门次数
    if controlling_team == 1:
        team_a.shots_on_net += 1
    else:
        team_b.shots_on_net += 1
    # 随机生成击中区域
    hit_area = randint(1, 5)
    # 如果射门区域等于击中区域，调用handle_hit函数处理进球
    if goal_area == hit_area:
        hit_area = handle_hit(
            controlling_team,
            team_a,
            team_b,
            player_index,
            goal_player,
            goal_assistant1,
            goal_assistant2,
            hit_area,
            z,
        )
    # 如果射门区域不等于击中区域，调用handle_miss函数处理未进球情况
    if goal_area != hit_area:
        return handle_miss(
            controlling_team, team_a, team_b, remaining_time, goal_player
        )
    # 打印信息并返回继续比赛和剩余时间
    print("AND WE'RE READY FOR THE FACE-OFF\n")
    return ("continue", remaining_time)
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```