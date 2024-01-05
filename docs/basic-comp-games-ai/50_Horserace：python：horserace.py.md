# `d:/src/tocomm/basic-computer-games\50_Horserace\python\horserace.py`

```
import math  # 导入 math 模块，用于数学运算
import random  # 导入 random 模块，用于生成随机数
import time  # 导入 time 模块，用于时间相关操作
from typing import List, Tuple  # 从 typing 模块中导入 List 和 Tuple 类型

def basic_print(*zones, **kwargs) -> None:
    """Simulates the PRINT command from BASIC to some degree.
    Supports `printing zones` if given multiple arguments."""
    # 定义一个函数 basic_print，用于模拟 BASIC 中的 PRINT 命令
    # *zones 表示接受任意数量的位置参数，**kwargs 表示接受任意数量的关键字参数

    line = ""  # 初始化一个空字符串 line
    if len(zones) == 1:  # 如果传入的位置参数数量为 1
        line = str(zones[0])  # 将第一个位置参数转换为字符串赋值给 line
    else:  # 如果传入的位置参数数量大于 1
        line = "".join([f"{str(zone):<14}" for zone in zones])  # 将每个位置参数转换为字符串并按照指定格式拼接成一行字符串
    identation = kwargs.get("indent", 0)  # 从关键字参数中获取 indent 的值，如果不存在则默认为 0
    end = kwargs.get("end", "\n")  # 从关键字参数中获取 end 的值，如果不存在则默认为换行符
    print(" " * identation + line, end=end)  # 打印带有缩进的字符串，结尾使用指定的结束符号
def basic_input(prompt: str, type_conversion=None):
    """BASIC INPUT command with optional type conversion"""
    # 定义一个函数，用于接收用户输入，并可选地进行类型转换
    while True:
        try:
            inp = input(f"{prompt}? ")  # 提示用户输入，并存储用户输入的值
            if type_conversion is not None:  # 如果有指定类型转换函数
                inp = type_conversion(inp)  # 对用户输入的值进行类型转换
            break  # 跳出循环
        except ValueError:  # 如果发生值错误
            basic_print("INVALID INPUT!")  # 打印错误信息
    return inp  # 返回用户输入的值


# horse names do not change over the program, therefore making it a global.
# throught the game, the ordering of the horses is used to indentify them
HORSE_NAMES = [
    "JOE MAW",  # 马的名字
    "L.B.J.",  # 马的名字
    "MR.WASHBURN",  # 马的名字
    ...
    "MISS KAREN",  # 添加选手名字"MISS KAREN"
    "JOLLY",  # 添加选手名字"JOLLY"
    "HORSE",  # 添加选手名字"HORSE"
    "JELLY DO NOT",  # 添加选手名字"JELLY DO NOT"
    "MIDNIGHT",  # 添加选手名字"MIDNIGHT"
]


def introduction() -> None:
    """Print the introduction, and optional the instructions"""

    basic_print("HORSERACE", indent=31)  # 打印标题"HORSERACE"，并设置缩进为31
    basic_print("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY", indent=15)  # 打印信息"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并设置缩进为15
    basic_print("\n\n")  # 打印两个换行符
    basic_print("WELCOME TO SOUTH PORTLAND HIGH RACETRACK")  # 打印欢迎信息"WELCOME TO SOUTH PORTLAND HIGH RACETRACK"
    basic_print("                      ...OWNED BY LAURIE CHEVALIER")  # 打印信息"...OWNED BY LAURIE CHEVALIER"
    y_n = basic_input("DO YOU WANT DIRECTIONS")  # 获取用户输入，询问是否需要指引

    # if no instructions needed, return
    if y_n.upper() == "NO":  # 如果用户输入的是"NO"，则返回
        return
```
这行代码是一个函数的返回语句，表示函数的执行结束并返回一个空值。

```
    basic_print("UP TO 10 MAY PLAY.  A TABLE OF ODDS WILL BE PRINTED.  YOU")
    basic_print("MAY BET ANY + AMOUNT UNDER 100000 ON ONE HORSE.")
    basic_print("DURING THE RACE, A HORSE WILL BE SHOWN BY ITS")
    basic_print("NUMBER.  THE HORSES RACE DOWN THE PAPER!")
    basic_print("")
```
这些代码是用来打印游戏规则和提示信息的，通过basic_print函数将文本信息输出到控制台。

```
def setup_players() -> List[str]:
    """Gather the number of players and their names"""

    # ensure we get an integer value from the user
    number_of_players = basic_input("HOW MANY WANT TO BET", int)

    # for each user query their name and return the list of names
    player_names = []
    basic_print("WHEN ? APPEARS,TYPE NAME")
    for _ in range(number_of_players):
        player_names.append(basic_input(""))
```
这部分代码是一个函数，用于收集玩家数量和他们的名字。首先通过basic_input函数获取玩家数量，然后通过循环依次获取每个玩家的名字，并将名字添加到player_names列表中。最后返回包含玩家名字的列表。
    return player_names
# 返回 player_names 变量的值

def setup_horses() -> List[float]:
    """Generates random odds for each horse. Returns a list of
    odds, indexed by the order of the global HORSE_NAMES."""
    # 为每匹马生成随机赔率。返回一个按全局变量 HORSE_NAMES 顺序索引的赔率列表

    odds = [random.randrange(1, 10) for _ in HORSE_NAMES]
    # 生成一个包含每匹马随机赔率的列表
    total = sum(odds)
    # 计算所有赔率的总和

    # rounding odds to two decimals for nicer output,
    # this is not in the origin implementation
    return [round(total / odd, 2) for odd in odds]
    # 将每个赔率四舍五入到两位小数，以便更好的输出。这不是原始实现中的部分

def print_horse_odds(odds) -> None:
    """Print the odds for each horse"""
    # 打印每匹马的赔率

    basic_print("")
    # 调用 basic_print 函数打印空行
    for i in range(len(HORSE_NAMES)):
    # 遍历全局变量 HORSE_NAMES 的长度
    # 打印每匹马的名称、编号和赔率
    basic_print(HORSE_NAMES[i], i, f"{odds[i]}:1")
    # 打印空行
    basic_print("")

def get_bets(player_names: List[str]) -> List[Tuple[int, float]]:
    """对于每个玩家，获取要下注的马匹编号和下注金额"""

    # 打印分隔线
    basic_print("--------------------------------------------------")
    # 打印提示信息
    basic_print("PLACE YOUR BETS...HORSE # THEN AMOUNT")

    # 初始化下注列表
    bets: List[Tuple[int, float]] = []
    # 遍历每个玩家
    for name in player_names:
        # 获取玩家下注的马匹编号
        horse = basic_input(name, int)
        # 初始化下注金额为None
        amount = None
        # 当下注金额为None时，循环获取有效的下注金额
        while amount is None:
            # 获取玩家输入的下注金额
            amount = basic_input("", float)
            # 如果下注金额小于1或大于等于100000，则提示玩家无法这样下注，并将下注金额设为None
            if amount < 1 or amount >= 100000:
                basic_print("  YOU CAN'T DO THAT!")
                amount = None
        bets.append((horse, amount))  # 将马名和下注金额添加到下注列表中

    basic_print("")  # 打印空行

    return bets  # 返回下注列表


def get_distance(odd: float) -> int:
    """Advances a horse during one step of the racing simulation.
    The amount travelled is random, but scaled by the odds of the horse"""

    d = random.randrange(1, 100)  # 生成1到100之间的随机数，表示马在一步中前进的距离
    s = math.ceil(odd)  # 将赔率向上取整，表示马的赔率
    if d < 10:  # 如果随机数小于10
        return 1  # 返回1
    elif d < s + 17:  # 如果随机数小于赔率加17
        return 2  # 返回2
    elif d < s + 37:  # 如果随机数小于赔率加37
        return 3  # 返回3
    elif d < s + 57:  # 如果随机数小于赔率加57
        return 4  # 如果条件成立，返回4
    elif d < s + 77:  # 如果条件成立，执行以下代码
        return 5  # 如果条件成立，返回5
    elif d < s + 92:  # 如果条件成立，执行以下代码
        return 6  # 如果条件成立，返回6
    else:  # 如果以上条件都不成立，执行以下代码
        return 7  # 返回7


def print_race_state(total_distance, race_pos) -> None:
    """Outputs the current state/stop of the race.
    Each horse is placed according to the distance they have travelled. In
    case some horses travelled the same distance, their numbers are printed
    on the same name"""

    # we dont want to modify the `race_pos` list, since we need
    # it later. Therefore we generating an interator from the list
    race_pos_iter = iter(race_pos)  # 从列表中生成一个迭代器，以便后续使用

    # race_pos is stored by last to first horse in the race.
    # 获取下一匹需要打印出来的马
    next_pos = next(race_pos_iter)

    # 起跑线
    basic_print("XXXXSTARTXXXX")

    # 打印赛道的所有 28 行/单位
    for line in range(28):

        # 确保我们仍有需要打印的马，并且如果有的话，检查下一匹需要打印的马是否不是当前行
        # 需要迭代，因为多匹马可以共享同一行
        while next_pos is not None and line == total_distance[next_pos]:
            basic_print(f"{next_pos} ", end="")
            next_pos = next(race_pos_iter, None)
        else:
            # 如果没有剩下的马需要打印这一行，打印一个新行
            basic_print("")

    # 终点线
    basic_print("XXXXFINISHXXXX")  # 打印出"XXXXFINISHXXXX"

def simulate_race(odds) -> List[int]:
    num_horses = len(HORSE_NAMES)  # 获取马匹名称列表的长度，即马匹的数量

    # in spirit of the original implementation, using two arrays to
    # track the total distance travelled, and create an index from
    # race position -> horse index
    total_distance = [0] * num_horses  # 创建一个长度为马匹数量的数组，用于跟踪每匹马的总距离

    # race_pos maps from the position in the race, to the index of the horse
    # it will later be sorted from last to first horse, based on the
    # distance travelled by each horse.
    # e.g. race_pos[0] => last horse
    #      race_pos[-1] => winning horse
    race_pos = list(range(num_horses))  # 创建一个从0到马匹数量-1的列表，用于表示比赛中每匹马的位置

    basic_print("\n1 2 3 4 5 6 7 8")  # 打印出"1 2 3 4 5 6 7 8"
    while True:  # 无限循环，表示比赛持续进行

        # advance each horse by a random amount
        for i in range(num_horses):  # 遍历每匹马
            total_distance[i] += get_distance(odds[i])  # 根据每匹马的赔率获取其前进的距离

        # bubble sort race_pos based on total distance travelled
        # in the original implementation, race_pos is reset for each
        # simulation step, so we keep this behaviour here
        race_pos = list(range(num_horses))  # 初始化赛马的位置列表
        for line in range(num_horses):  # 遍历每匹马
            for i in range(num_horses - 1 - line):  # 遍历每匹马的位置
                if total_distance[race_pos[i]] < total_distance[race_pos[i + 1]]:  # 如果前一匹马的距离小于后一匹马的距离
                    continue  # 继续下一次循环
                race_pos[i], race_pos[i + 1] = race_pos[i + 1], race_pos[i]  # 交换位置，使得距离较大的马排在前面

        # print current state of the race
        print_race_state(total_distance, race_pos)  # 打印当前比赛状态

        # goal line is defined as 28 units from start
        # 检查获胜的马是否已经越过终点线
        if total_distance[race_pos[-1]] >= 28:
            return race_pos

        # 这不是原始的BASIC实现中的内容，但它可以使比赛可视化成一个漂亮的动画（如果终端大小设置为31行）
        time.sleep(1)


def print_race_results(race_positions, odds, bets, player_names) -> None:
    """打印比赛结果，以及每个玩家的赢利"""

    # 首先打印比赛位置
    basic_print("THE RACE RESULTS ARE:")
    for position, horse_idx in enumerate(reversed(race_positions), start=1):
        line = f"{position} PLACE HORSE NO. {horse_idx} AT {odds[horse_idx]}:1"
        basic_print("")
        basic_print(line)

    # 接着打印玩家赢得的金额
    winning_horse_idx = race_positions[-1]  # 获取比赛结果中最后一匹马的索引
    for idx, name in enumerate(player_names):  # 遍历玩家名称列表，同时获取索引
        (horse, amount) = bets[idx]  # 获取玩家下注的马匹和金额
        if horse == winning_horse_idx:  # 如果玩家下注的马匹与获胜的马匹相同
            basic_print("")  # 打印空行
            basic_print(f"{name} WINS ${amount * odds[winning_horse_idx]}")  # 打印玩家赢得的金额

def main_loop(player_names, horse_odds) -> None:
    """Main game loop"""

    while True:  # 进入游戏主循环
        print_horse_odds(horse_odds)  # 打印马匹赔率
        bets = get_bets(player_names)  # 获取玩家的下注
        final_race_positions = simulate_race(horse_odds)  # 模拟比赛结果
        print_race_results(final_race_positions, horse_odds, bets, player_names)  # 打印比赛结果和玩家下注情况

        basic_print("DO YOU WANT TO BET ON THE NEXT RACE ?")  # 打印是否想要在下一场比赛下注
        one_more = basic_input("YES OR NO")  # 获取玩家输入的是否继续下注
        if one_more.upper() != "YES":  # 如果玩家输入不是YES，则退出循环
            break  # 终止当前循环，跳出循环体

# 定义主函数
def main() -> None:
    # introduction, player names and horse odds are only generated once
    introduction()  # 调用introduction函数，输出游戏介绍
    player_names = setup_players()  # 调用setup_players函数，设置玩家名称并返回玩家名称列表
    horse_odds = setup_horses()  # 调用setup_horses函数，设置马匹赔率并返回赔率字典

    # main loop of the game, the player can play multiple races, with the
    # same odds
    main_loop(player_names, horse_odds)  # 调用main_loop函数，进行游戏主循环，玩家可以进行多次比赛，赔率不变

# 如果当前脚本为主程序，则执行main函数
if __name__ == "__main__":
    main()
```