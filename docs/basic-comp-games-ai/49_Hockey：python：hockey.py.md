# `d:/src/tocomm/basic-computer-games\49_Hockey\python\hockey.py`

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

@dataclass  # 使用 dataclass 装饰器定义一个类
class Team:  # 定义一个名为 Team 的类
    # TODO: It would be better to use a Player-class (name, goals, assits)
    #       and have the attributes directly at each player. This would avoid
    #       dealing with indices that much
    #
    #       I'm also rather certain that I messed up somewhere with the indices
    #       - instead of using those, one could use actual player positions:
    #       LEFT WING,    CENTER,        RIGHT WING
    #       LEFT DEFENSE, RIGHT DEFENSE, GOALKEEPER
    name: str  # 定义一个字符串类型的变量name，用于存储球队名称
    players: List[str]  # 6 players  # 定义一个列表类型的变量players，用于存储球队的6名球员
    shots_on_net: int = 0  # 定义一个整数类型的变量shots_on_net，用于存储射门次数，默认值为0
    goals: List[int] = field(default_factory=lambda: [0 for _ in range(NB_PLAYERS)])  # 定义一个列表类型的变量goals，用于存储球员的进球数，默认值为0
    assists: List[int] = field(default_factory=lambda: [0 for _ in range(NB_PLAYERS)])  # 定义一个列表类型的变量assists，用于存储球员的助攻数，默认值为0
    score: int = 0  # 定义一个整数类型的变量score，用于存储球队的得分，默认值为0

    def show_lineup(self) -> None:  # 定义一个方法show_lineup，用于展示球队的阵容
        print(" " * 10 + f"{self.name} STARTING LINEUP")  # 打印球队名称和STARTING LINEUP
        for player in self.players:  # 遍历球队的球员
            print(player)  # 打印每个球员的信息
# 定义一个函数，用于询问用户是否输入二进制数据
def ask_binary(prompt: str, error_msg: str) -> bool:
    while True:
        answer = input(prompt).lower()  # 提示用户输入信息，并将输入转换为小写
        if answer in ["y", "yes"]:  # 如果用户输入是"y"或"yes"
            return True  # 返回True
        if answer in ["n", "no"]:  # 如果用户输入是"n"或"no"
            return False  # 返回False
        print(error_msg)  # 打印错误信息


# 定义一个函数，用于获取两个团队的名称
def get_team_names() -> Tuple[str, str]:
    while True:
        answer = input("ENTER THE TWO TEAMS: ")  # 提示用户输入两个团队的名称
        if answer.count(",") == 1:  # 如果输入中包含一个逗号
            return answer.split(",")  # type: ignore  # 以逗号为分隔符，将输入的团队名称分割成一个包含两个元素的元组并返回
        print("separated by a single comma")  # 打印错误信息


# 定义一个函数，用于获取密码
def get_pass() -> int:
    while True:  # 创建一个无限循环，直到条件满足才会退出循环
        answer = input("PASS? ")  # 提示用户输入一个值
        try:  # 尝试执行以下代码
            passes = int(answer)  # 将用户输入的值转换为整数
            if passes >= 0 and passes <= 3:  # 如果输入的值在0到3之间
                return passes  # 返回输入的值
        except ValueError:  # 如果出现值错误
            print("ENTER A NUMBER BETWEEN 0 AND 3")  # 提示用户输入一个0到3之间的数字


def get_minutes_per_game() -> int:  # 定义一个函数，返回一个整数
    while True:  # 创建一个无限循环，直到条件满足才会退出循环
        answer = input("ENTER THE NUMBER OF MINUTES IN A GAME ")  # 提示用户输入一个值
        try:  # 尝试执行以下代码
            minutes = int(answer)  # 将用户输入的值转换为整数
            if minutes >= 1:  # 如果输入的值大于等于1
                return minutes  # 返回输入的值
        except ValueError:  # 如果出现值错误
            print("ENTER A NUMBER")  # 提示用户输入一个数字
# 定义一个函数，用于获取球员的名字列表
def get_player_names(prompt: str) -> List[str]:
    players = []  # 创建一个空列表，用于存储球员名字
    print(prompt)  # 打印提示信息
    for i in range(1, 7):  # 循环6次，获取6个球员的名字
        player = input(f"PLAYER {i}: ")  # 获取球员名字
        players.append(player)  # 将球员名字添加到列表中
    return players  # 返回球员名字列表


# 定义一个函数，用于进行投篮
def make_shot(
    controlling_team: int, team_a: Team, team_b: Team, player_index: List[int], j: int
) -> Tuple[int, int, int, int]:
    while True:  # 进入无限循环
        try:
            s = int(input("SHOT? "))  # 获取用户输入的投篮数
        except ValueError:  # 如果输入不是整数
            continue  # 继续循环
        if s >= 1 and s <= 4:  # 如果投篮数在1到4之间
            break  # 退出循环
    if controlling_team == 1:  # 如果控制的是1队
        print(team_a.players[player_index[j - 1]])  # 打印1队的球员信息
    else:
        print(team_b.players[player_index[j - 1]])  # 否则打印2队的球员信息
    g = player_index[j - 1]  # 将球员索引赋值给g
    g1 = 0  # 初始化g1为0
    g2 = 0  # 初始化g2为0
    if s == 1:  # 如果s等于1
        print(" LET'S A BOOMER GO FROM THE RED LINE!!\n")  # 打印特定信息
        z = 10  # 将z赋值为10
    elif s == 2:  # 如果s等于2
        print(" FLIPS A WRISTSHOT DOWN THE ICE\n")  # 打印特定信息
        # 可能原始代码缺少了第430行
    elif s == 3:  # 如果s等于3
        print(" BACKHANDS ONE IN ON THE GOALTENDER\n")  # 打印特定信息
        z = 25  # 将z赋值为25
    elif s == 4:  # 如果s等于4
        print(" SNAPS A LONG FLIP SHOT\n")  # 打印特定信息
        z = 17  # 将z赋值为17
    return z, g, g1, g2  # 返回变量 z, g, g1, g2 的值


def print_header() -> None:
    print(" " * 33 + "HOCKEY")  # 打印标题 "HOCKEY"
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")  # 打印创意计算和地点信息


def instructions() -> None:
    wants_it = ask_binary("WOULD YOU LIKE THE INSTRUCTIONS? ", "ANSWER YES OR NO!!")  # 询问用户是否需要游戏说明
    if wants_it:  # 如果用户需要说明
        print()  # 打印空行
        print("THIS IS A SIMULATED HOCKEY GAME.")  # 打印游戏说明
        print("QUESTION     RESPONSE")  # 打印问题和回答
        print("PASS         TYPE IN THE NUMBER OF PASSES YOU WOULD")  # 打印传球说明
        print("             LIKE TO MAKE, FROM 0 TO 3.")
        print("SHOT         TYPE THE NUMBER CORRESPONDING TO THE SHOT")  # 打印射门说明
        print("             YOU WANT TO MAKE.  ENTER:")
        print("             1 FOR A SLAPSHOT")
        print("             2 FOR A WRISTSHOT")
        print("             3 FOR A BACKHAND")  # 打印提示信息，提示用户输入3表示进行反手击球
        print("             4 FOR A SNAP SHOT")  # 打印提示信息，提示用户输入4表示进行快速射门
        print("AREA         TYPE IN THE NUMBER CORRESPONDING TO")  # 打印提示信息，提示用户输入数字对应的区域
        print("             THE AREA YOU ARE AIMING AT.  ENTER:")  # 打印提示信息，提示用户输入瞄准的区域
        print("             1 FOR UPPER LEFT HAND CORNER")  # 打印提示信息，提示用户输入1表示瞄准左上角
        print("             2 FOR UPPER RIGHT HAND CORNER")  # 打印提示信息，提示用户输入2表示瞄准右上角
        print("             3 FOR LOWER LEFT HAND CORNER")  # 打印提示信息，提示用户输入3表示瞄准左下角
        print("             4 FOR LOWER RIGHT HAND CORNER")  # 打印提示信息，提示用户输入4表示瞄准右下角
        print("AT THE START OF THE GAME, YOU WILL BE ASKED FOR THE NAMES")  # 打印提示信息，提示游戏开始时需要输入球员的名字
        print("OF YOUR PLAYERS.  THEY ARE ENTERED IN THE ORDER: ")  # 打印提示信息，提示球员的输入顺序
        print("LEFT WING, CENTER, RIGHT WING, LEFT DEFENSE,")  # 打印提示信息，提示输入左翼、中锋、右翼、左后卫
        print("RIGHT DEFENSE, GOALKEEPER.  ANY OTHER INPUT REQUIRED WILL")  # 打印提示信息，提示输入右后卫、守门员
        print("HAVE EXPLANATORY INSTRUCTIONS.")  # 打印提示信息，提示还会有其他说明性的指令
            team_a.players[player_index[j - 2]]  # 获取球队A中球员索引为j-2的球员
            + " LEADS "  # 添加字符串 " LEADS "
            + team_a.players[player_index[j - 1]]  # 获取球队A中球员索引为j-1的球员
            + " WITH A PERFECT PASS.\n"  # 添加字符串 " WITH A PERFECT PASS." 并换行
        )
        print(team_a.players[player_index[j - 1]] + " CUTTING IN!!!\n")  # 打印球队A中球员索引为j-1的球员的动作
        scoring_player = player_index[j - 1]  # 将球员索引为j-1的球员设为得分球员
        goal_assistant1 = player_index[j - 2]  # 将球员索引为j-2的球员设为助攻球员1
        goal_assistant2 = 0  # 将助攻球员2设为0
        z1 = 3  # 将z1设为3
    elif pass_value == 2:  # 如果传球值为2
        print(
            team_a.players[player_index[j - 2]]  # 打印球队A中球员索引为j-2的球员
            + " GIVES TO A STREAKING "  # 添加字符串 " GIVES TO A STREAKING "
            + team_a.players[player_index[j - 1]]  # 获取球队A中球员索引为j-1的球员
        )
        print(
            team_a.players[player_index[j - 3]]  # 获取球队A中球员索引为j-3的球员
            + " COMES DOWN ON "  # 添加字符串 " COMES DOWN ON "
            + team_b.players[4]  # 获取球队B中索引为4的球员
        + " AND "  # 连接字符串，用于打印输出
        + team_b.players[3]  # 获取第四个球员的信息，用于打印输出
    )
    scoring_player = player_index[j - 3]  # 获取得分球员的索引
    goal_assistant1 = player_index[j - 1]  # 获取助攻球员1的索引
    goal_assistant2 = player_index[j - 2]  # 获取助攻球员2的索引
    z1 = 2  # 设置变量z1的值为2
elif pass_value == 3:  # 如果传球值为3
    print("OH MY GOD!! A ' 4 ON 2 ' SITUATION\n")  # 打印输出特定信息
    print(
        team_a.players[player_index[j - 3]]  # 打印输出特定球员信息
        + " LEADS "
        + team_a.players[player_index[j - 2]]  # 打印输出特定球员信息
        + "\n"
    )
    print(team_a.players[player_index[j - 2]] + " IS WHEELING THROUGH CENTER.\n")  # 打印输出特定球员信息
    print(
        team_a.players[player_index[j - 2]]  # 打印输出特定球员信息
        + " GIVES AND GOEST WITH "
        + team_a.players[player_index[j - 1]]  # 打印输出特定球员信息
        )
        # 打印“PRETTY PASSING!”
        print("PRETTY PASSING!\n")
        # 打印球队A的球员将球传给球队A的另一名球员
        print(
            team_a.players[player_index[j - 1]]
            + " DROPS IT TO "
            + team_a.players[player_index[j - 4]]
        )
        # 记录得分球员的索引
        scoring_player = player_index[j - 4]
        # 记录助攻球员1的索引
        goal_assistant1 = player_index[j - 1]
        # 记录助攻球员2的索引
        goal_assistant2 = player_index[j - 2]
        # 初始化z1为1
        z1 = 1
    # 返回得分球员、助攻球员1、助攻球员2和z1
    return scoring_player, goal_assistant1, goal_assistant2, z1


# 定义team2_action函数，接受传球值、球员索引列表、球队A、球队B和j作为参数，返回一个元组
def team2_action(
    pass_value: int, player_index: List[int], team_a: Team, team_b: Team, j: int
) -> Tuple[int, int, int, int]:
    # 如果传球值为1
    if pass_value == 1:
        # 打印球队B的球员将球传给球队B的另一名球员
        print(
            team_b.players[player_index[j - 1]]
scoring_player = player_index[j - 2]  # 保存得分球员的索引
goal_assistant1 = player_index[j - 1]  # 保存助攻球员1的索引
goal_assistant2 = 0  # 初始化助攻球员2的索引为0
z1 = 3  # 初始化z1为3
elif pass_value == 2:  # 如果传球值为2
    print("IT'S A ' 3 ON 2 '!\n")  # 打印信息表明是3打2的情况
    print(
        "ONLY " + team_a.players[3] + " AND " + team_a.players[4] + " ARE BACK.\n"
    )  # 打印信息表明只有球队A的第3和第4号球员在防守
    print(
        team_b.players[player_index[j - 2]]
        + " GIVES OFF TO "
        + team_b.players[player_index[j - 1]]
    )  # 打印信息表明球员j-2传球给了球员j-1
    print(
        team_b.players[player_index[j - 1]]  # 打印球员j-1的信息
        + " DROPS TO "
        + team_b.players[player_index[j - 3]]
    )
    # 设置得分球员的索引
    scoring_player = player_index[j - 3]
    # 设置助攻球员1的索引
    goal_assistant1 = player_index[j - 1]
    # 设置助攻球员2的索引
    goal_assistant2 = player_index[j - 2]
    # 设置 z1 的值为 2
    z1 = 2
elif pass_value == 3:
    # 打印特定情况下的描述
    print(" A '3 ON 2 ' WITH A ' TRAILER '!\n")
    print(
        team_b.players[player_index[j - 4]]
        + " GIVES TO "
        + team_b.players[player_index[j - 2]]
        + " WHO SHUFFLES IT OFF TO\n"
    )
    print(
        team_b.players[player_index[j - 1]] + " WHO FIRES A WING TO WING PASS TO \n"
    )
    print(team_b.players[player_index[j - 3]] + " AS HE CUTS IN ALONE!!\n")
    # 设置得分球员的索引
    scoring_player = player_index[j - 3]
        goal_assistant1 = player_index[j - 1]  # 获取进球助攻1的球员索引
        goal_assistant2 = player_index[j - 2]  # 获取进球助攻2的球员索引
        z1 = 1  # 设置z1的值为1
    return scoring_player, goal_assistant1, goal_assistant2, z1  # 返回得分球员、进球助攻1、进球助攻2、z1的值


def final_message(team_a: Team, team_b: Team, player_index: List[int]) -> None:
    # Bells chime  # 钟声响起
    print("THAT'S THE SIREN\n")  # 打印“这就是警报声”
    print("\n")  # 打印空行
    print(" " * 15 + "FINAL SCORE:\n")  # 打印最终比分
    if team_b.score <= team_a.score:  # 如果B队得分小于等于A队得分
        print(f"{team_a.name}: {team_a.score}\t{team_b.name}: {team_b.score}\n")  # 打印A队得分和B队得分
    else:  # 否则
        print(f"{team_b.name}: {team_b.score}\t{team_a.name}\t:{team_a.score}\n")  # 打印B队得分和A队得分
    print("\n")  # 打印空行
    print(" " * 10 + "SCORING SUMMARY\n")  # 打印得分总结
    print("\n")  # 打印空行
    print(" " * 25 + team_a.name + "\n")  # 打印A队名称
    print("\tNAME\tGOALS\tASSISTS\n")  # 打印姓名、进球数、助攻数
    # 打印分隔线
    print("\t----\t-----\t-------\n")
    # 遍历并打印team_a的球员、进球数和助攻数
    for i in range(1, 6):
        print(f"\t{team_a.players[i]}\t{team_a.goals[i]}\t{team_a.assists[i]}\n")
    # 打印空行
    print("\n")
    # 打印team_b的名称
    print(" " * 25 + team_b.name + "\n")
    # 打印表头
    print("\tNAME\tGOALS\tASSISTS\n")
    # 打印分隔线
    print("\t----\t-----\t-------\n")
    # 遍历并打印team_b的球员、进球数和助攻数
    for t in range(1, 6):
        print(f"\t{team_b.players[t]}\t{team_b.goals[t]}\t{team_b.assists[t]}\n")
    # 打印空行
    print("\n")
    # 打印射门次数
    print("SHOTS ON NET\n")
    # 打印team_a的射门次数
    print(f"{team_a.name}: {team_a.shots_on_net}\n")
    # 打印team_b的射门次数
    print(team_b.name + f": {team_b.shots_on_net}\n")


def main() -> None:
    # 打印介绍信息
    print_header()
    # 创建一个长度为21的整数列表并初始化为0
    player_index: List[int] = [0 for _ in range(21)]
    # 打印3个空行
    print("\n" * 3)
    instructions()  # 调用 instructions() 函数，显示游戏指令

    # Gather input
    team_name_a, team_name_b = get_team_names()  # 调用 get_team_names() 函数，获取两支球队的名称
    print()
    minutes_per_game = get_minutes_per_game()  # 调用 get_minutes_per_game() 函数，获取比赛每节的分钟数
    print()
    players_a = get_player_names(f"WOULD THE {team_name_a} COACH ENTER HIS TEAM")  # 调用 get_player_names() 函数，获取第一支球队的球员名单
    print()
    players_b = get_player_names(f"WOULD THE {team_name_b} COACH DO THE SAME")  # 调用 get_player_names() 函数，获取第二支球队的球员名单
    team_a = Team(team_name_a, players_a)  # 创建第一支球队的 Team 对象
    team_b = Team(team_name_b, players_b)  # 创建第二支球队的 Team 对象
    print()
    referee = input("INPUT THE REFEREE FOR THIS GAME: ")  # 获取裁判的姓名
    print()
    team_a.show_lineup()  # 显示第一支球队的阵容
    print()
    team_b.show_lineup()  # 显示第二支球队的阵容
    print("WE'RE READY FOR TONIGHTS OPENING FACE-OFF.")  # 打印准备好进行今晚的开场比赛
    print(  # 打印
        f"{referee} WILL DROP THE PUCK BETWEEN "
        f"{team_a.players[0]} AND {team_b.players[0]}"
    )
    # 设置剩余比赛时间为每场比赛的分钟数
    remaining_time = minutes_per_game

    # 进行比赛
    while remaining_time > 0:
        # 模拟比赛回合，并更新剩余时间
        cont, remaining_time = simulate_game_round(
            team_a, team_b, player_index, remaining_time
        )
        remaining_time -= 1
        # 如果比赛回合结果为"break"，则跳出循环
        if cont == "break":
            break

    # 结束语
    final_message(team_a, team_b, player_index)


def handle_hit(
    controlling_team: int,
    team_a: Team,  # 参数：代表球队A的对象
    team_b: Team,  # 参数：代表球队B的对象
    player_index: List[int],  # 参数：代表球员索引的列表
    goal_player: int,  # 参数：代表进球球员的索引
    goal_assistant1: int,  # 参数：代表助攻球员1的索引
    goal_assistant2: int,  # 参数：代表助攻球员2的索引
    hit_area: int,  # 参数：代表射门区域的索引
    z: int,  # 参数：代表一个整数

) -> int:  # 返回类型注释：返回一个整数
    while True:  # 进入无限循环
        player_index[20] = randint(1, 100)  # 将球员索引列表中的第21个位置赋值为1到100之间的随机整数
        if player_index[20] % z != 0:  # 如果第21个位置的值除以z的余数不等于0
            break  # 退出循环
        a2 = randint(1, 100)  # 生成1到100之间的随机整数赋值给a2
        if a2 % 4 == 0:  # 如果a2除以4的余数等于0
            if controlling_team == 1:  # 如果控球队为1
                print(f"SAVE {team_b.players[5]} --  REBOUND\n")  # 打印保存球员B的信息和篮板
            else:  # 否则
                print(f"SAVE {team_a.players[5]} --  FOLLOW up\n")  # 打印保存球员A的信息和跟进
            continue  # 继续下一次循环
        else:
            hit_area += 1  # 如果条件不满足，则增加 hit_area 的值
    if player_index[20] % z != 0:  # 如果 player_index[20] 除以 z 的余数不等于 0
        if controlling_team == 1:  # 如果控制球队为 1
            print(f"GOAL {team_a.name}\n")  # 打印进球信息
            team_a.score += 1  # 球队 A 的得分加一
        else:
            print(f"SCORE {team_b.name}\n")  # 打印得分信息
            team_b.score += 1  # 球队 B 的得分加一
        # Bells in origninal
        print("\n")  # 打印空行
        print("SCORE: ")  # 打印 "SCORE: "
        if team_b.score <= team_a.score:  # 如果球队 B 的得分小于等于球队 A 的得分
            print(f"{team_a.name}: {team_a.score}\t{team_b.name}: {team_b.score}\n")  # 打印球队 A 和 B 的得分
        else:
            print(f"{team_b.name}: {team_b.score}\t{team_a.name}: {team_a.score}\n")  # 打印球队 B 和 A 的得分
        if controlling_team == 1:  # 如果控制球队为 1
            team = team_a  # 将 team 赋值为球队 A
        else:
            team = team_b  # 将 team 赋值为球队 B
        # 打印进球球员的信息
        print("GOAL SCORED BY: " + team.players[goal_player] + "\n")
        # 如果有第一助攻球员
        if goal_assistant1 != 0:
            # 如果有第二助攻球员
            if goal_assistant2 != 0:
                # 打印两名助攻球员的信息
                print(
                    f" ASSISTED BY: {team.players[goal_assistant1]}"
                    f" AND {team.players[goal_assistant2]}"
                )
            else:
                # 打印第一名助攻球员的信息
                print(f" ASSISTED BY: {team.players[goal_assistant1]}")
            # 给助攻球员增加助攻次数
            team.assists[goal_assistant1] += 1
            team.assists[goal_assistant2] += 1
        else:
            # 打印无人助攻的信息
            print(" UNASSISTED.\n")
        # 给进球球员增加进球次数
        team.goals[goal_player] += 1

    # 返回击球区域
    return hit_area


def handle_miss(
    controlling_team: int,
    team_a: Team,  # 参数 team_a 是 Team 类型的对象
    team_b: Team,  # 参数 team_b 是 Team 类型的对象
    remaining_time: int,  # 参数 remaining_time 是整数类型
    goal_player: int,  # 参数 goal_player 是整数类型
) -> Tuple[str, int]:  # 返回类型为元组，包含一个字符串和一个整数

    saving_player = randint(1, 7)  # 生成一个 1 到 7 之间的随机整数，表示扑救球员的编号

    if controlling_team == 1:  # 如果控球队是 1
        if saving_player == 1:  # 如果扑救球员编号为 1
            print("KICK SAVE AND A BEAUTY BY " + team_b.players[5] + "\n")  # 打印扑救动作和扑救球员的名字
            print("CLEARED OUT BY " + team_b.players[3] + "\n")  # 打印清除球员的名字
            remaining_time -= 1  # 剩余时间减 1
            return ("continue", remaining_time)  # 返回一个包含字符串 "continue" 和剩余时间的元组
        if saving_player == 2:  # 如果扑救球员编号为 2
            print("WHAT A SPECTACULAR GLOVE SAVE BY " + team_b.players[5] + "\n")  # 打印扑救动作和扑救球员的名字
            print("AND " + team_b.players[5] + " GOLFS IT INTO THE CROWD\n")  # 打印球员将球打向观众席的动作
            return ("break", remaining_time)  # 返回一个包含字符串 "break" 和剩余时间的元组
        if saving_player == 3:  # 如果扑救球员编号为 3
            print("SKATE SAVE ON A LOW STEAMER BY " + team_b.players[5] + "\n")  # 打印扑救动作和扑救球员的名字
            remaining_time -= 1  # 剩余时间减 1
            return ("continue", remaining_time)  # 返回一个包含字符串 "continue" 和剩余时间的元组
        # 如果守门员编号为4，打印球队B的第5号球员挡出球，并且球队A的进攻球员和球队B的第5号球员将球控制住
        if saving_player == 4:
            print(f"PAD SAVE BY {team_b.players[5]} OFF THE STICK\n")
            print(
                f"OF {team_a.players[goal_player]} AND "
                f"{team_b.players[5]} COVERS UP\n"
            )
            return ("break", remaining_time)
        # 如果守门员编号为5，打印球队B的第5号球员头顶挡出球，并且剩余时间减1
        if saving_player == 5:
            print(f"WHISTLES ONE OVER THE HEAD OF {team_b.players[5]}\n")
            remaining_time -= 1
            return ("continue", remaining_time)
        # 如果守门员编号为6，打印球队B的第5号球员做出了面部挡出，并且受伤，然后球队B的防守球员控制住球
        if saving_player == 6:
            print(f"{team_b.players[5]} MAKES A FACE SAVE!! AND HE IS HURT\n")
            print(f"THE DEFENSEMAN {team_b.players[5]} COVERS UP FOR HIM\n")
            return ("break", remaining_time)
    else:
        # 如果守门员编号为1，打印球队A的第5号球员挡出球，并且球队A的第3号球员将球清出
        if saving_player == 1:
            print(f"STICK SAVE BY {team_a.players[5]}\n")
            print(f"AND CLEARED OUT BY {team_a.players[3]}\n")
            remaining_time -= 1
        if saving_player == 1:  # 如果扑救球员是1
            return ("continue", remaining_time)  # 返回继续比赛和剩余时间
        if saving_player == 2:  # 如果扑救球员是2
            print(  # 打印以下内容
                "OH MY GOD!! "
                f"{team_b.players[goal_player]} RATTLES ONE OFF THE POST\n"
            )
            print(  # 打印以下内容
                f"TO THE RIGHT OF {team_a.players[5]} AND "
                f"{team_a.players[5]} COVERS "
            )
            print("ON THE LOOSE PUCK!\n")  # 打印"ON THE LOOSE PUCK!\n"
            return ("break", remaining_time)  # 返回中断比赛和剩余时间
        if saving_player == 3:  # 如果扑救球员是3
            print("SKATE SAVE BY " + team_a.players[5] + "\n")  # 打印"SKATE SAVE BY " + team_a.players[5] + "\n"
            print(team_a.players[5] + " WHACKS THE LOOSE PUCK INTO THE STANDS\n")  # 打印team_a.players[5] + " WHACKS THE LOOSE PUCK INTO THE STANDS\n"
            return ("break", remaining_time)  # 返回中断比赛和剩余时间
        if saving_player == 4:  # 如果扑救球员是4
            print(  # 打印以下内容
                "STICK SAVE BY " + team_a.players[5] + " AND HE CLEARS IT OUT HIMSELF\n"
            )
            remaining_time -= 1  # 减少剩余时间
            return ("continue", remaining_time)  # 返回继续比赛的状态和剩余时间
        if saving_player == 5:  # 如果救球员是5号球员
            print("KICKED OUT BY " + team_a.players[5] + "\n")  # 打印被5号球员踢出
            print("AND IT REBOUNDS ALL THE WAY TO CENTER ICE\n")  # 打印球弹到中心冰面
            remaining_time -= 1  # 减少剩余时间
            return ("continue", remaining_time)  # 返回继续比赛的状态和剩余时间
        if saving_player == 6:  # 如果救球员是6号球员
            print("GLOVE SAVE " + team_a.players[5] + " AND HE HANGS ON\n")  # 打印6号球员扑出球并抓住
            return ("break", remaining_time)  # 返回暂停比赛的状态和剩余时间
    return ("continue", remaining_time)  # 返回继续比赛的状态和剩余时间


def simulate_game_round(
    team_a: Team, team_b: Team, player_index: List[int], remaining_time: int
) -> Tuple[str, int]:  # 模拟比赛回合，接收两支球队、球员索引列表和剩余时间，返回状态和剩余时间的元组
    controlling_team = randint(1, 2)  # 随机确定控球队伍
    if controlling_team == 1:  # 如果控球队伍是1号队伍
        print(f"{team_a.name} HAS CONTROL OF THE PUCK.")  # 打印1号队伍控球
    else:  # 如果控球队伍是2号队伍
        print(f"{team_b.name} HAS CONTROL.")  # 打印出球队 B 控球的信息
    pass_value = get_pass()  # 调用 get_pass() 函数获取传球值，并赋值给 pass_value
    for i in range(1, 4):  # 循环遍历 i 从 1 到 3
        player_index[i] = 0  # 将 player_index[i] 的值设为 0

    # Line 310:
    while True:  # 进入无限循环
        j = 0  # 初始化 j 为 0
        for j in range(1, (pass_value + 2) + 1):  # 循环遍历 j 从 1 到 pass_value + 2 + 1
            player_index[j] = randint(1, 5)  # 为 player_index[j] 赋一个 1 到 5 之间的随机整数
        if player_index[j - 1] == player_index[j - 2] or (
            pass_value + 2 >= 3
            and (
                player_index[j - 1] == player_index[j - 3]
                or player_index[j - 2] == player_index[j - 3]
            )
        ):  # 如果满足条件则跳出循环
            break
    if pass_value == 0:  # line 350  # 如果传球值为 0
        z, goal_player, goal_assistant1, goal_assistant2 = make_shot(  # 调用 make_shot() 函数
        controlling_team, team_a, team_b, player_index, j
    )
else:
    if controlling_team == 1:
        # 调用team1_action函数，传入参数pass_value, player_index, team_a, team_b, j，并将返回值赋给goal_player, goal_assistant1, goal_assistant2, z1
        goal_player, goal_assistant1, goal_assistant2, z1 = team1_action(
            pass_value, player_index, team_a, team_b, j
        )
    else:
        # 调用team2_action函数，传入参数pass_value, player_index, team_a, team_b, j，并将返回值赋给goal_player, goal_assistant1, goal_assistant2, z1
        goal_player, goal_assistant1, goal_assistant2, z1 = team2_action(
            pass_value, player_index, team_a, team_b, j
        )
    while True:
        # 循环直到输入的shot_type不小于1且不大于4
        shot_type = int(input("SHOT? "))
        if not (shot_type < 1 or shot_type > 4):
            break
    if controlling_team == 1:
        # 打印team_a中goal_player位置的球员
        print(team_a.players[goal_player], end="")
    else:
        # 打印team_b中goal_player位置的球员
        print(team_b.players[goal_player], end="")
    if shot_type == 1:
            # 打印“LET'S A BIG SLAP SHOT GO!!\n”
            print(" LET'S A BIG SLAP SHOT GO!!\n")
            # 初始化变量z为4
            z = 4
            # 将z1的值加到z上
            z += z1
        # 如果射门类型为2
        if shot_type == 2:
            # 打印“RIPS A WRIST SHOT OFF\n”
            print(" RIPS A WRIST SHOT OFF\n")
            # 初始化变量z为2
            z = 2
            # 将z1的值加到z上
            z += z1
        # 如果射门类型为3
        if shot_type == 3:
            # 打印“GETS A BACKHAND OFF\n”
            print(" GETS A BACKHAND OFF\n")
            # 初始化变量z为3
            z = 3
            # 将z1的值加到z上
            z += z1
        # 如果射门类型为4
        if shot_type == 4:
            # 打印“SNAPS OFF A SNAP SHOT\n”
            print(" SNAPS OFF A SNAP SHOT\n")
            # 初始化变量z为2
            z = 2
            # 将z1的值加到z上
            z += z1
    # 循环直到输入的goal_area不小于1且不大于4
    while True:
        # 从用户输入中获取goal_area的值
        goal_area = int(input("AREA? "))
        # 如果goal_area不小于1且不大于4，则跳出循环
        if not (goal_area < 1 or goal_area > 4):
            break
    # 如果控制球队为1
    if controlling_team == 1:
    # 如果进球区域等于命中区域，调用处理进球的函数
    if goal_area == hit_area:
        hit_area = handle_hit(
            controlling_team,  # 控球队
            team_a,  # A队
            team_b,  # B队
            player_index,  # 球员索引
            goal_player,  # 进球球员
            goal_assistant1,  # 助攻球员1
            goal_assistant2,  # 助攻球员2
            hit_area,  # 命中区域
            z,  # z值
        )
    # 如果进球区域不等于命中区域，调用处理未进球的函数
    if goal_area != hit_area:
        return handle_miss(
            controlling_team,  # 控球队
            team_a,  # A队
            team_b,  # B队
            remaining_time,  # 剩余时间
            goal_player  # 进球球员
        )
    print("AND WE'RE READY FOR THE FACE-OFF\n")  # 打印出"AND WE'RE READY FOR THE FACE-OFF"，并换行
    return ("continue", remaining_time)  # 返回一个元组，包含字符串"continue"和变量remaining_time的值


if __name__ == "__main__":
    main()  # 如果当前脚本被直接执行，则调用main函数
```