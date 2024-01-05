# `d:/src/tocomm/basic-computer-games\27_Civil_War\python\Civilwar.py`

```
"""
Original game design: Cram, Goodie, Hibbard Lexington H.S.
Modifications: G. Paul, R. Hess (Ties), 1973
"""
# 导入所需的模块
import enum  # 用于创建枚举类型
import math  # 提供数学运算函数
import random  # 生成随机数
from dataclasses import dataclass  # 用于创建数据类
from typing import Dict, List, Literal, Tuple  # 用于类型提示


# 定义攻击状态的枚举类型
class AttackState(enum.Enum):
    DEFENSIVE = 1
    BOTH_OFFENSIVE = 2
    OFFENSIVE = 3


# 定义常量
CONF = 1
UNION = 2
@dataclass
class PlayerStat:
    # 定义玩家状态的数据结构，包括食物、工资、弹药等属性
    food: float = 0
    salaries: float = 0
    ammunition: float = 0

    # 定义玩家状态的数据结构，包括逃兵、伤亡、士气、战略等属性
    desertions: float = 0
    casualties: float = 0
    morale: float = 0
    strategy: int = 0
    available_men: int = 0
    available_money: int = 0

    # 定义玩家状态的数据结构，包括陆军和海军的属性
    army_c: float = 0
    army_m: float = 0  # available_men ????
    inflation: float = 0

    # 定义玩家状态的数据结构，包括 r 和 t 属性
    r: float = 0
    t: float = 0  # casualties + desertions
    q: float = 0  # 累积成本？
    p: float = 0  # 累积利润？
    m: float = 0  # 累积资金？

    is_player = False  # 是否为玩家
    excessive_losses = False  # 是否有过多损失

    def set_available_money(self):  # 设置可用资金
        if self.is_player:  # 如果是玩家
            factor = 1 + (self.r - self.q) / (self.r + 1)  # 计算因子
        else:
            factor = 1  # 否则因子为1
        self.available_money = 100 * math.floor(  # 计算可用资金
            (self.army_m * (100 - self.inflation) / 2000) * factor + 0.5
        )

    def get_cost(self) -> float:  # 获取成本
        return self.food + self.salaries + self.ammunition  # 返回食物、工资和弹药的总和作为成本

    def get_army_factor(self) -> float:  # 获取军队因子
        return 1 + (self.p - self.t) / (self.m + 1)  # 计算并返回一个数值

    def get_present_men(self) -> float:  # 返回一个浮点数
        return self.army_m * self.get_army_factor()  # 返回军队数量乘以军队因素

def simulate_losses(player1: PlayerStat, player2: PlayerStat) -> float:
    """Simulate losses of player 1"""
    tmp = (2 * player1.army_c / 5) * (  # 计算临时值
        1 + 1 / (2 * (abs(player1.strategy - player2.strategy) + 1))  # 根据玩家策略计算值
    )
    tmp = tmp * (1.28 + (5 * player1.army_m / 6) / (player1.ammunition + 1))  # 根据军队数量和弹药计算值
    tmp = math.floor(tmp * (1 + 1 / player1.morale) + 0.5)  # 根据士气计算值并向下取整
    return tmp  # 返回计算结果


def update_army(player: PlayerStat, enemy: PlayerStat, use_factor=False) -> None:
    player.casualties = simulate_losses(player, enemy)  # 更新玩家的伤亡人数
    player.desertions = 100 / player.morale  # 更新玩家的叛逃人数
    # 计算损失人数，包括战争造成的伤亡和士兵的叛变
    loss = player.casualties + player.desertions
    # 如果不使用因子，将当前可用人数设为浮点数
    if not use_factor:
        present_men: float = player.available_men
    # 否则，调用获取当前可用人数的方法
    else:
        present_men = player.get_present_men()
    # 如果损失人数大于等于当前可用人数
    if loss >= present_men:
        # 获取军队因子
        factor = player.get_army_factor()
        # 如果不使用因子，将因子设为1
        if not use_factor:
            factor = 1
        # 计算伤亡人数，向下取整
        player.casualties = math.floor(13 * player.army_m / 20 * factor)
        # 计算叛变人数
        player.desertions = 7 * player.casualties / 13
        # 标记为过度损失
        player.excessive_losses = True


def get_choice(prompt: str, choices: List[str]) -> str:
    # 循环直到输入的选择在给定的选项中
    while True:
        choice = input(prompt)
        if choice in choices:
            break
    return choice
def get_morale(stat: PlayerStat, enemy: PlayerStat) -> float:
    """Higher is better"""  # 函数的目的是计算士气值，返回值类型为浮点数
    enemy_strength = 5 * enemy.army_m / 6  # 计算敌方军队的力量
    return (2 * math.pow(stat.food, 2) + math.pow(stat.salaries, 2)) / math.pow(
        enemy_strength, 2
    ) + 1  # 返回计算得到的士气值


def main() -> None:
    battles = [  # 创建一个包含多个战斗信息的列表
        [
            "JULY 21, 1861.  GEN. BEAUREGARD, COMMANDING THE SOUTH, MET",
            "UNION FORCES WITH GEN. MCDOWELL IN A PREMATURE BATTLE AT",
            "BULL RUN. GEN. JACKSON HELPED PUSH BACK THE UNION ATTACK.",
        ],
        [
            "APRIL 6-7, 1862.  THE CONFEDERATE SURPRISE ATTACK AT",
            "SHILOH FAILED DUE TO POOR ORGANIZATION.",  # 添加战斗信息到列表
        ],
        [
            "JUNE 25-JULY 1, 1862.  GENERAL LEE (CSA) UPHELD THE",
            "OFFENSIVE THROUGHOUT THE BATTLE AND FORCED GEN. MCCLELLAN",
            "AND THE UNION FORCES AWAY FROM RICHMOND.",
        ],
        [
            "AUG 29-30, 1862.  THE COMBINED CONFEDERATE FORCES UNDER LEE",
            "AND JACKSON DROVE THE UNION FORCES BACK INTO WASHINGTON.",
        ],
        [
            "SEPT 17, 1862.  THE SOUTH FAILED TO INCORPORATE MARYLAND",
            "INTO THE CONFEDERACY.",
        ],
        [
            "DEC 13, 1862.  THE CONFEDERACY UNDER LEE SUCCESSFULLY",
            "REPULSED AN ATTACK BY THE UNION UNDER GEN. BURNSIDE.",
        ],
        ["DEC 31, 1862.  THE SOUTH UNDER GEN. BRAGG WON A CLOSE BATTLE."],
        [
```

这部分代码是一个包含多个列表的列表，每个内部列表包含了一些字符串。
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 以二进制模式打开文件，读取文件内容，封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，'r'表示以只读模式打开 ZIP 文件
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名列表，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
historical_data: List[Tuple[str, float, float, float, int, AttackState]] = [
        # 创建一个包含元组的列表，每个元组包含战争名称、部队数量、伤亡人数、俘虏人数、攻击状态
        ("", 0, 0, 0, 0, AttackState.DEFENSIVE),  # 空战争名称，部队数量、伤亡人数、俘虏人数均为0，攻击状态为防御
        ("BULL RUN", 18000, 18500, 1967, 2708, AttackState.DEFENSIVE),  # 战争名称为BULL RUN，部队数量、伤亡人数、俘虏人数分别为18000、18500、1967，攻击状态为防御
        ("SHILOH", 40000.0, 44894.0, 10699, 13047, AttackState.OFFENSIVE),  # 战争名称为SHILOH，部队数量、伤亡人数、俘虏人数分别为40000.0、44894.0、10699，攻击状态为进攻
        ("SEVEN DAYS", 95000.0, 115000.0, 20614, 15849, AttackState.OFFENSIVE),  # 战争名称为SEVEN DAYS，部队数量、伤亡人数、俘虏人数分别为95000.0、115000.0、20614，攻击状态为进攻
        ("SECOND BULL RUN", 54000.0, 63000.0, 10000, 14000, AttackState.BOTH_OFFENSIVE),  # 战争名称为SECOND BULL RUN，部队数量、伤亡人数、俘虏人数分别为54000.0、63000.0、10000，攻击状态为双方均进攻
        ("ANTIETAM", 40000.0, 50000.0, 10000, 12000, AttackState.OFFENSIVE),  # 战争名称为ANTIETAM，部队数量、伤亡人数、俘虏人数分别为40000.0、50000.0、10000，攻击状态为进攻
        ("FREDERICKSBURG", 75000.0, 120000.0, 5377, 12653, AttackState.DEFENSIVE),  # 战争名称为FREDERICKSBURG，部队数量、伤亡人数、俘虏人数分别为75000.0、120000.0、5377，攻击状态为防御
        ("MURFREESBORO", 38000.0, 45000.0, 11000, 12000, AttackState.DEFENSIVE),  # 战争名称为MURFREESBORO，部队数量、伤亡人数、俘虏人数分别为38000.0、45000.0、11000，攻击状态为防御
        ("CHANCELLORSVILLE", 32000, 90000.0, 13000, 17197, AttackState.BOTH_OFFENSIVE),  # 战争名称为CHANCELLORSVILLE，部队数量、伤亡人数、俘虏人数分别为32000、90000.0、13000，攻击状态为双方均进攻
        ("VICKSBURG", 50000.0, 70000.0, 12000, 19000, AttackState.DEFENSIVE),  # 战争名称为VICKSBURG，部队数量、伤亡人数、俘虏人数分别为50000.0、70000.0、12000，攻击状态为防御
    ]
        ("GETTYSBURG", 72500.0, 85000.0, 20000, 23000, AttackState.OFFENSIVE), 
        # 创建一个包含战争信息的元组，包括地点名称、联邦军和南方联盟军的兵力、炮火和攻击状态
        ("CHICKAMAUGA", 66000.0, 60000.0, 18000, 16000, AttackState.BOTH_OFFENSIVE),
        ("CHATTANOOGA", 37000.0, 60000.0, 36700.0, 5800, AttackState.BOTH_OFFENSIVE),
        ("SPOTSYLVANIA", 62000.0, 110000.0, 17723, 18000, AttackState.BOTH_OFFENSIVE),
        ("ATLANTA", 65000.0, 100000.0, 8500, 3700, AttackState.DEFENSIVE),
    ]
    confederate_strategy_prob_distribution = {}
    # 创建一个空的字典，用于存储南方联盟军的战略概率分布

    # What do you spend money on?
    stats: Dict[int, PlayerStat] = {
        CONF: PlayerStat(),
        UNION: PlayerStat(),
    }
    # 创建一个包含玩家统计信息的字典，包括联邦军和南方联盟军的统计信息

    print(" " * 26 + "CIVIL WAR")
    # 打印"CIVIL WAR"，并在前面添加26个空格

    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    # 打印"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并在前面添加15个空格

    # Union info on likely confederate strategy
    confederate_strategy_prob_distribution[1] = 25
    # 将南方联盟军的战略概率分布中1号位置的概率设置为25
    confederate_strategy_prob_distribution[2] = 25
    # 将南方联盟军的战略概率分布中2号位置的概率设置为25
    # 设置联邦军策略概率分布的第三个元素为25
    confederate_strategy_prob_distribution[3] = 25
    # 设置联邦军策略概率分布的第四个元素为25
    confederate_strategy_prob_distribution[4] = 25
    # 打印空行
    print()
    # 获取用户选择是否需要游戏说明
    show_instructions = get_choice(
        "DO YOU WANT INSTRUCTIONS? YES OR NO -- ", ["YES", "NO"]
    )

    # 如果用户选择需要游戏说明
    if show_instructions == "YES":
        # 打印多行游戏说明
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
        print("POSSIBLE.")  # 打印可能性提示
        print()  # 打印空行
        print("YOUR CHOICES FOR DEFENSIVE STRATEGY ARE:")  # 打印防御策略选择提示
        print("        (1) ARTILLERY ATTACK")  # 打印防御策略选项1
        print("        (2) FORTIFICATION AGAINST FRONTAL ATTACK")  # 打印防御策略选项2
        print("        (3) FORTIFICATION AGAINST FLANKING MANEUVERS")  # 打印防御策略选项3
        print("        (4) FALLING BACK")  # 打印防御策略选项4
        print(" YOUR CHOICES FOR OFFENSIVE STRATEGY ARE:")  # 打印进攻策略选择提示
        print("        (1) ARTILLERY ATTACK")  # 打印进攻策略选项1
        print("        (2) FRONTAL ATTACK")  # 打印进攻策略选项2
        print("        (3) FLANKING MANEUVERS")  # 打印进攻策略选项3
        print("        (4) ENCIRCLEMENT")  # 打印进攻策略选项4
        print("YOU MAY SURRENDER BY TYPING A '5' FOR YOUR STRATEGY.")  # 提示可以通过输入5来投降

    print()  # 打印空行
    print()  # 打印空行
    print()  # 打印空行
    print("ARE THERE TWO GENERALS PRESENT ", end="")  # 打印询问是否有两位将军在场
    two_generals = get_choice("(ANSWER YES OR NO) ", ["YES", "NO"]) == "YES"  # 获取用户输入并将结果存储在变量two_generals中
    stats[CONF].is_player = True  # 将stats[CONF].is_player设置为True
    if two_generals:  # 如果条件为真
        party: Literal[1, 2] = 2  # 将 party 设置为 2，表示游戏中的玩家数量
        stats[UNION].is_player = True  # 将 stats[UNION].is_player 设置为 True
    else:  # 如果条件为假
        party = 1  # 将 party 设置为 1
        print()  # 打印空行
        print("YOU ARE THE CONFEDERACY.   GOOD LUCK!")  # 打印提示信息
        print()  # 打印空行

    print("SELECT A BATTLE BY TYPING A NUMBER FROM 1 TO 14 ON")  # 打印提示信息
    print("REQUEST.  TYPE ANY OTHER NUMBER TO END THE SIMULATION.")  # 打印提示信息
    print("BUT '0' BRINGS BACK EXACT PREVIOUS BATTLE SITUATION")  # 打印提示信息
    print("ALLOWING YOU TO REPLAY IT")  # 打印提示信息
    print()  # 打印空行
    print("NOTE: A NEGATIVE FOOD$ ENTRY CAUSES THE PROGRAM TO ")  # 打印提示信息
    print("USE THE ENTRIES FROM THE PREVIOUS BATTLE")  # 打印提示信息
    print()  # 打印空行
    print("AFTER REQUESTING A BATTLE, DO YOU WISH ", end="")  # 打印提示信息
    print("BATTLE DESCRIPTIONS ", end="")  # 打印提示信息
    xs = get_choice("(ANSWER YES OR NO) ", ["YES", "NO"])  # 获取用户选择
    # 初始化南北战争统计数据
    confederacy_lost = 0  # 南方失败次数
    confederacy_win = 0   # 南方胜利次数
    for i in [CONF, UNION]:  # 遍历南北方
        stats[i].p = 0  # 初始化人口
        stats[i].m = 0  # 初始化军队数量
        stats[i].t = 0  # 初始化战争次数
        stats[i].available_money = 0  # 初始化可用资金
        stats[i].food = 0  # 初始化食物
        stats[i].salaries = 0  # 初始化薪水
        stats[i].ammunition = 0  # 初始化弹药
        stats[i].strategy = 0  # 初始化战略
        stats[i].excessive_losses = False  # 初始化是否有过多损失
    confederacy_unresolved = 0  # 未解决的南方战争次数
    random_nb: float = 0  # 随机数
    while True:  # 进入循环
        print()  # 打印空行
        print()  # 打印空行
        print()  # 打印空行
        simulated_battle_index = int(  # 获取模拟战斗索引
            get_choice(
        "WHICH BATTLE DO YOU WISH TO SIMULATE? (0-14) ",  # 提示用户输入要模拟的战役编号
        [str(i) for i in range(15)],  # 创建一个包含0到14的字符串列表，用于用户选择
    )
    if simulated_battle_index < 1 or simulated_battle_index > 14:  # 如果用户输入的战役编号不在0到14之间
        break  # 结束循环
    if simulated_battle_index != 0 or random_nb == 0:  # 如果用户选择的战役编号不是0或者随机数为0
        loaded_battle = historical_data[simulated_battle_index]  # 从历史数据中加载用户选择的战役信息
        battle_name = loaded_battle[0]  # 获取战役名称
        stats[CONF].army_m = loaded_battle[1]  # 设置联盟军的士兵数量
        stats[UNION].army_m = loaded_battle[2]  # 设置联邦军的士兵数量
        stats[CONF].army_c = loaded_battle[3]  # 设置联盟军的城市数量
        stats[UNION].army_c = loaded_battle[4]  # 设置联邦军的城市数量
        stats[CONF].excessive_losses = False  # 设置联盟军的损失为False

        # Inflation calc
        stats[CONF].inflation = 10 + (confederacy_lost - confederacy_win) * 2  # 计算联盟军的通货膨胀
        stats[UNION].inflation = 10 + (confederacy_win - confederacy_lost) * 2  # 计算联邦军的通货膨胀

        # Money and Men available
            # 遍历列表 [CONF, UNION]
            for i in [CONF, UNION]:
                # 调用 stats 对象的 set_available_money 方法，设置可用资金
                stats[i].set_available_money()
                # 将 stats 对象的 available_men 属性设置为 stats 对象的 get_army_factor 方法返回值的向下取整
                stats[i].available_men = math.floor(stats[i].get_army_factor())
            # 打印空行
            print()
            print()
            print()
            print()
            print()
            # 打印战斗名称
            print(f"THIS IS THE BATTLE OF {battle_name}")
            # 如果 xs 不等于 "NO"，则打印模拟战斗的内容
            if xs != "NO":
                print("\n".join(battles[simulated_battle_index - 1]))

        else:
            # 打印战斗名称的即时重播
            print(f"{battle_name} INSTANT REPLAY")

        # 打印空行
        print()
        # 打印 CONFEDERACY 和 UNION 的可用士兵数量
        print("          CONFEDERACY\t UNION")
        print(f"MEN       {stats[CONF].available_men}\t\t {stats[UNION].available_men}")
        # 打印 CONFEDERACY 和 UNION 的可用资金
        print(
            f"MONEY    ${stats[CONF].available_money}\t${stats[UNION].available_money}"
        )
        # 打印输出联邦和联盟的通货膨胀率
        print(f"INFLATION {stats[CONF].inflation + 15}%\t\t {stats[UNION].inflation}%")
        print()
        # 只有在打印输出中，CONFED的通货膨胀率才会增加15%
        # 如果有两位将军，先输入CONFED
        for player_index in range(1, party + 1):
            if two_generals and player_index == 1:
                print("CONFEDERATE GENERAL---", end="")
            print("HOW MUCH DO YOU WISH TO SPEND FOR")
            # 循环直到输入合法的食物数量
            while True:
                food_input = int(input(" - FOOD...... ? "))
                if food_input < 0:
                    if stats[CONF].r == 0:
                        print("NO PREVIOUS ENTRIES")
                        continue
                    print("ASSUME YOU WANT TO KEEP SAME ALLOCATIONS")
                    print()
                    break
                stats[player_index].food = food_input
                while True:
                    # 为指定球员索引的球员对象设置薪水属性，要求用户输入整数值
                    stats[player_index].salaries = int(input(" - SALARIES.. ? "))
                    # 如果薪水值大于等于0，则跳出循环
                    if stats[player_index].salaries >= 0:
                        break
                    # 如果薪水值小于0，则打印错误信息并继续循环
                    print("NEGATIVE VALUES NOT ALLOWED.")
                # 循环直到用户输入非负整数为止
                while True:
                    # 为指定球员索引的球员对象设置弹药属性，要求用户输入整数值
                    stats[player_index].ammunition = int(input(" - AMMUNITION ? "))
                    # 如果弹药值大于等于0，则跳出循环
                    if stats[player_index].ammunition >= 0:
                        break
                    # 如果弹药值小于0，则打印错误信息并继续循环
                    print("NEGATIVE VALUES NOT ALLOWED.")
                # 打印空行
                print()
                # 如果球员对象的花费大于可用资金，则打印警告信息
                if stats[player_index].get_cost() > stats[player_index].available_money:
                    print(
                        f"THINK AGAIN! YOU HAVE ONLY ${stats[player_index].available_money}"
                    )
                # 否则跳出循环
                else:
                    break

            # 如果没有两位将军或者当前球员索引为2，则跳出循环
            if not two_generals or player_index == 2:
                break
            # 否则打印信息并继续循环
            print("UNION GENERAL---", end="")
        for player_index in range(1, party + 1):  # 遍历玩家索引，从1到party
            if two_generals:  # 如果有两位将军
                if player_index == 1:  # 如果玩家索引为1
                    print("CONFEDERATE ", end="")  # 打印"CONFEDERATE "，不换行
                else:  # 否则
                    print("      UNION ", end="")  # 打印"      UNION "，不换行
            morale = get_morale(stats[player_index], stats[1 + player_index % 2])  # 获取玩家的士气值

            if morale >= 10:  # 如果士气值大于等于10
                print("MORALE IS HIGH")  # 打印"MORALE IS HIGH"
            elif morale >= 5:  # 否则如果士气值大于等于5
                print("MORALE IS FAIR")  # 打印"MORALE IS FAIR"
            else:  # 否则
                print("MORALE IS POOR")  # 打印"MORALE IS POOR"
            if not two_generals:  # 如果没有两位将军
                break  # 跳出循环
            stats[player_index].morale = morale  # type: ignore  # 将玩家的士气值赋给stats[player_index].morale

        stats[UNION].morale = get_morale(stats[UNION], stats[CONF])  # 获取UNION的士气值
        stats[CONF].morale = get_morale(stats[CONF], stats[UNION])  # 设置CONF方的士气值，根据UNION方的情况计算
        print("CONFEDERATE GENERAL---")  # 打印提示信息
        # Actual off/def battle situation
        if loaded_battle[5] == AttackState.OFFENSIVE:  # 如果战斗状态为进攻
            print("YOU ARE ON THE OFFENSIVE")  # 打印提示信息
        elif loaded_battle[5] == AttackState.DEFENSIVE:  # 如果战斗状态为防守
            print("YOU ARE ON THE DEFENSIVE")  # 打印提示信息
        else:  # 如果战斗状态为其他
            print("BOTH SIDES ARE ON THE OFFENSIVE")  # 打印提示信息

        print()  # 打印空行
        # Choose strategies
        if not two_generals:  # 如果没有两位将军
            while True:  # 进入循环
                stats[CONF].strategy = int(input("YOUR STRATEGY "))  # 获取用户输入的战略选择
                if abs(stats[CONF].strategy - 3) < 3:  # 如果战略选择合法
                    break  # 退出循环
                print(f"STRATEGY {stats[CONF].strategy} NOT ALLOWED.")  # 打印提示信息
            if stats[CONF].strategy == 5:  # 如果战略选择为5
                print("THE CONFEDERACY HAS SURRENDERED.")  # 打印提示信息
                break  # 结束当前循环

            # Union strategy is computer chosen
            if simulated_battle_index == 0:  # 如果模拟战斗索引为0
                while True:  # 进入无限循环
                    stats[UNION].strategy = int(input("UNION STRATEGY IS "))  # 从用户输入中获取联盟策略
                    if stats[UNION].strategy > 0 and stats[UNION].strategy < 5:  # 如果联盟策略大于0且小于5
                        break  # 结束循环
                    print("ENTER 1, 2, 3, OR 4 (USUALLY PREVIOUS UNION STRATEGY)")  # 打印提示信息
            else:  # 否则
                s0 = 0  # 初始化s0为0
                random_nb = random.random() * 100  # 生成一个0-100之间的随机数
                for player_index in range(1, 5):  # 遍历1到4的范围
                    s0 += confederate_strategy_prob_distribution[player_index]  # 累加联盟策略概率分布
                    # If actual strategy info is in program data statements
                    # then r-100 is extra weight given to that strategy.
                    if random_nb < s0:  # 如果随机数小于s0
                        break  # 结束循环
                stats[UNION].strategy = player_index  # 设置联盟策略为player_index
                print(stats[UNION].strategy)  # 打印联盟策略
        else:  # 否则
# 遍历玩家索引列表，包括1和2
for player_index in [1, 2]:
    # 如果玩家索引为1，打印提示信息
    if player_index == 1:
        print("CONFEDERATE STRATEGY ? ", end="")
    # 无限循环，直到满足条件跳出循环
    while True:
        # 从输入中获取CONF玩家的策略，并转换为整数
        stats[CONF].strategy = int(input())
        # 如果CONF玩家的策略与3的差的绝对值小于3，跳出循环
        if abs(stats[CONF].strategy - 3) < 3:
            break
        # 打印不允许的策略信息
        print(f"STRATEGY {stats[CONF].strategy} NOT ALLOWED.")
        print("YOUR STRATEGY ? ", end="")
    # 如果玩家索引为2
    if player_index == 2:
        # 将UNION玩家的策略设置为CONF玩家的策略
        stats[UNION].strategy = stats[CONF].strategy
        # 将CONF玩家的策略设置为之前保存的策略
        stats[CONF].strategy = previous_strategy  # type: ignore # noqa: F821
        # 如果UNION玩家的策略不等于5，跳出循环
        if stats[UNION].strategy != 5:
            break
    else:
        # 保存CONF玩家的策略
        previous_strategy = stats[CONF].strategy  # noqa: F841
    # 打印提示信息
    print("UNION STRATEGY ? ", end="")

# 更新军队信息
update_army(stats[UNION], stats[CONF], use_factor=False)
        # 计算模拟损失
        print()
        print()
        print()
        print("\t\tCONFEDERACY\tUNION")
        # 更新军队统计数据，使用因子
        update_army(stats[CONF], stats[UNION], use_factor=True)

        if party == 1:
            # 计算联盟方的伤亡人数
            stats[UNION].casualties = math.floor(
                17
                * stats[UNION].army_c
                * stats[CONF].army_c
                / (stats[CONF].casualties * 20)
            )
            # 计算联盟方的叛逃人数
            stats[CONF].desertions = 5 * morale

        # 打印输出联盟方和联邦方的伤亡人数
        print(
            "CASUALTIES\t"
            + str(stats[CONF].casualties)
            + "\t\t"
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
# 打印联盟方的损失占原始军队数量的百分比
print(
    "UNION:       "
    + str(
        math.floor(
            100 * (stats[UNION].casualties / stats[UNION].army_c) + 0.5
        )
    )
    + "% OF THE ORIGINAL"
)

print()
# 查找谁赢了
if (
    stats[CONF].excessive_losses
    and stats[UNION].excessive_losses
    or (
        not stats[CONF].excessive_losses
        and not stats[UNION].excessive_losses
        and stats[CONF].casualties + stats[CONF].desertions
        == stats[UNION].casualties + stats[CONF].desertions
```
这段代码是用来打印联盟方的损失占原始军队数量的百分比，并且查找谁赢了。
            )
        ):
            # 如果战斗结果无法确定，则打印“战斗结果未解决”，并将联盟未解决战斗次数加一
            print("BATTLE OUTCOME UNRESOLVED")
            confederacy_unresolved += 1
        # 如果联盟方损失过多，或者（联盟方未损失过多且联邦方未损失过多且联盟方伤亡加上联盟方叛逃大于联邦方伤亡加上联邦方叛逃）
        elif stats[CONF].excessive_losses or (
            not stats[CONF].excessive_losses
            and not stats[UNION].excessive_losses
            and stats[CONF].casualties + stats[CONF].desertions
            > stats[UNION].casualties + stats[CONF].desertions
        ):
            # 打印“联邦方赢得了战斗名称”，如果模拟战斗索引不为0，则将联盟方失败次数加一
            print(f"THE UNION WINS {battle_name}")
            if simulated_battle_index != 0:
                confederacy_lost += 1
        else:
            # 打印“联盟方赢得了战斗名称”，如果模拟战斗索引不为0，则将联盟方胜利次数加一
            print(f"THE CONFEDERACY WINS {battle_name}")
            if simulated_battle_index != 0:
                confederacy_win += 1

        # 原始代码中的2530到2590行是无法到达的。
        if simulated_battle_index != 0:
            for i in [CONF, UNION]:
                # 更新统计数据
                stats[i].t += stats[i].casualties + stats[i].desertions
                stats[i].p += stats[i].army_c
                stats[i].q += stats[i].get_cost()
                stats[i].r += stats[i].army_m * (100 - stats[i].inflation) / 20
                stats[i].m += stats[i].army_m
            # 学习当前策略，开始遗忘旧策略
            # 南方的当前策略增加3*s，其他方失去s的概率点，除非某个策略下降到5%以下
            s = 3
            s0 = 0
            for player_index in range(1, 5):
                if confederate_strategy_prob_distribution[player_index] <= 5:
                    continue
                confederate_strategy_prob_distribution[player_index] -= 5
                s0 += s
            confederate_strategy_prob_distribution[stats[CONF].strategy] += s0

        # 重置南方和北方的 excessive_losses 属性为 False
        stats[CONF].excessive_losses = False
        stats[UNION].excessive_losses = False
        print("---------------")  # 打印分隔线

        continue  # 继续执行下一次循环

    print()  # 打印空行
    print()  # 打印空行
    print()  # 打印空行
    print()  # 打印空行
    print()  # 打印空行
    print()  # 打印空行
    print(
        f"THE CONFEDERACY HAS WON {confederacy_win} BATTLES AND LOST {confederacy_lost}"
    )  # 打印南方联盟赢得的战斗次数和输掉的战斗次数
    if stats[CONF].strategy == 5 or (
        stats[UNION].strategy != 5 and confederacy_win <= confederacy_lost
    ):
        print("THE UNION HAS WON THE WAR")  # 如果南方联盟的策略为5，或者北方联盟的策略不为5且南方联盟赢得的战斗次数小于等于输掉的战斗次数，则打印北方联盟赢得了战争
    else:
        print("THE CONFEDERACY HAS WON THE WAR")  # 否则打印南方联盟赢得了战争
    print()  # 打印空行
    if stats[CONF].r > 0:  # 如果南方联盟的资源大于0
        # 打印南北战争的战斗次数（不包括重新运行的）
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
        # 打印损失占原始数量的百分比
        print(
            f"    % OF ORIGINAL\t{math.floor(100 * (stats[CONF].t / stats[CONF].p) + 0.5)}\t{math.floor(100 * (stats[UNION].t / stats[UNION].p) + 0.5)}"
        )
        # 如果不是两位将军，则打印以下内容
        if not two_generals:
            print()
            print("UNION INTELLIGENCE SUGGEST THAT THE SOUTH USED")
            print("STRATEGIES 1, 2, 3, 4 IN THE FOLLOWING PERCENTAGES")
            print(
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制数据，封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```