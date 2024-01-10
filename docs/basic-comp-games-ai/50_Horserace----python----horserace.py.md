# `basic-computer-games\50_Horserace\python\horserace.py`

```
# 导入 math 模块
import math
# 导入 random 模块
import random
# 导入 time 模块
import time
# 导入 List 和 Tuple 类型
from typing import List, Tuple

# 定义一个模拟 BASIC PRINT 命令的函数，支持打印多个区域
def basic_print(*zones, **kwargs) -> None:
    """Simulates the PRINT command from BASIC to some degree.
    Supports `printing zones` if given multiple arguments."""

    # 初始化空字符串
    line = ""
    # 如果只有一个参数，将其转换为字符串
    if len(zones) == 1:
        line = str(zones[0])
    # 如果有多个参数，将它们连接成一个字符串
    else:
        line = "".join([f"{str(zone):<14}" for zone in zones])
    # 获取缩进值，默认为 0
    identation = kwargs.get("indent", 0)
    # 获取结束符，默认为换行符
    end = kwargs.get("end", "\n")
    # 打印带有缩进的字符串
    print(" " * identation + line, end=end)

# 定义一个模拟 BASIC INPUT 命令的函数，支持可选的类型转换
def basic_input(prompt: str, type_conversion=None):
    """BASIC INPUT command with optional type conversion"""

    # 循环直到输入合法
    while True:
        try:
            # 获取用户输入
            inp = input(f"{prompt}? ")
            # 如果需要类型转换，进行转换
            if type_conversion is not None:
                inp = type_conversion(inp)
            break
        # 捕获值错误异常
        except ValueError:
            # 提示输入无效
            basic_print("INVALID INPUT!")
    return inp

# 马的名字在整个程序中不会改变，因此将其定义为全局变量
HORSE_NAMES = [
    "JOE MAW",
    "L.B.J.",
    "MR.WASHBURN",
    "MISS KAREN",
    "JOLLY",
    "HORSE",
    "JELLY DO NOT",
    "MIDNIGHT",
]

# 打印介绍和可选的说明
def introduction() -> None:
    """Print the introduction, and optional the instructions"""

    # 打印标题
    basic_print("HORSERACE", indent=31)
    basic_print("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY", indent=15)
    basic_print("\n\n")
    basic_print("WELCOME TO SOUTH PORTLAND HIGH RACETRACK")
    basic_print("                      ...OWNED BY LAURIE CHEVALIER")
    # 获取用户输入，是否需要说明
    y_n = basic_input("DO YOU WANT DIRECTIONS")

    # 如果不需要说明，直接返回
    if y_n.upper() == "NO":
        return

    # 打印游戏说明
    basic_print("UP TO 10 MAY PLAY.  A TABLE OF ODDS WILL BE PRINTED.  YOU")
    basic_print("MAY BET ANY + AMOUNT UNDER 100000 ON ONE HORSE.")
    basic_print("DURING THE RACE, A HORSE WILL BE SHOWN BY ITS")
    basic_print("NUMBER.  THE HORSES RACE DOWN THE PAPER!")
    basic_print("")

# 设置玩家
def setup_players() -> List[str]:
    """Gather the number of players and their names"""

    # 从用户获取一个整数值
    number_of_players = basic_input("HOW MANY WANT TO BET", int)

    # 对于每个用户，询问他们的名字，并返回名字列表
    player_names = []
    basic_print("WHEN ? APPEARS,TYPE NAME")
    for _ in range(number_of_players):
        player_names.append(basic_input(""))
    # 返回名字列表
    return player_names
def setup_horses() -> List[float]:
    """Generates random odds for each horse. Returns a list of
    odds, indexed by the order of the global HORSE_NAMES."""

    # 为每匹马生成随机赔率，返回一个按全局变量 HORSE_NAMES 顺序索引的赔率列表
    odds = [random.randrange(1, 10) for _ in HORSE_NAMES]
    total = sum(odds)

    # 将赔率四舍五入到两位小数，以便更好的输出
    # 这不是原始实现中的一部分
    return [round(total / odd, 2) for odd in odds]


def print_horse_odds(odds) -> None:
    """Print the odds for each horse"""

    basic_print("")
    for i in range(len(HORSE_NAMES)):
        basic_print(HORSE_NAMES[i], i, f"{odds[i]}:1")
    basic_print("")


def get_bets(player_names: List[str]) -> List[Tuple[int, float]]:
    """For each player, get the number of the horse to bet on,
    as well as the amount of money to bet"""

    basic_print("--------------------------------------------------")
    basic_print("PLACE YOUR BETS...HORSE # THEN AMOUNT")

    bets: List[Tuple[int, float]] = []
    for name in player_names:
        horse = basic_input(name, int)
        amount = None
        while amount is None:
            amount = basic_input("", float)
            if amount < 1 or amount >= 100000:
                basic_print("  YOU CAN'T DO THAT!")
                amount = None
        bets.append((horse, amount))

    basic_print("")

    return bets


def get_distance(odd: float) -> int:
    """Advances a horse during one step of the racing simulation.
    The amount travelled is random, but scaled by the odds of the horse"""

    # 在赛马模拟的一步中使一匹马前进
    # 旅行的距离是随机的，但受到马的赔率的影响
    d = random.randrange(1, 100)
    s = math.ceil(odd)
    if d < 10:
        return 1
    elif d < s + 17:
        return 2
    elif d < s + 37:
        return 3
    elif d < s + 57:
        return 4
    elif d < s + 77:
        return 5
    elif d < s + 92:
        return 6
    else:
        return 7


def print_race_state(total_distance, race_pos) -> None:
    """Outputs the current state/stop of the race.
    Each horse is placed according to the distance they have travelled. In
    """
    # 如果有一些马匹走过了相同的距离，它们的编号会被打印在同一行上

    # 我们不想修改 `race_pos` 列表，因为我们之后还需要它。因此我们从列表中生成一个迭代器
    race_pos_iter = iter(race_pos)

    # race_pos 是按照比赛中最后一匹马到第一匹马的顺序存储的。
    # 我们获取需要打印出来的下一匹马
    next_pos = next(race_pos_iter)

    # 起跑线
    basic_print("XXXXSTARTXXXX")

    # 打印比赛赛道的所有 28 行/单位
    for line in range(28):

        # 确保我们仍然有需要打印的马匹，如果有的话，检查下一匹需要打印的马是否不是当前行
        # 需要迭代，因为多匹马可以共享同一行
        while next_pos is not None and line == total_distance[next_pos]:
            basic_print(f"{next_pos} ", end="")
            next_pos = next(race_pos_iter, None)
        else:
            # 如果没有马匹需要打印在这一行，打印一个新行
            basic_print("")

    # 终点线
    basic_print("XXXXFINISHXXXX")
def simulate_race(odds) -> List[int]:
    num_horses = len(HORSE_NAMES)  # 获取马匹数量

    # in spirit of the original implementation, using two arrays to
    # track the total distance travelled, and create an index from
    # race position -> horse index
    total_distance = [0] * num_horses  # 初始化每匹马的总距离为0

    # race_pos maps from the position in the race, to the index of the horse
    # it will later be sorted from last to first horse, based on the
    # distance travelled by each horse.
    # e.g. race_pos[0] => last horse
    #      race_pos[-1] => winning horse
    race_pos = list(range(num_horses))  # 初始化赛马位置列表

    basic_print("\n1 2 3 4 5 6 7 8")  # 打印赛道的起始位置

    while True:

        # advance each horse by a random amount
        for i in range(num_horses):
            total_distance[i] += get_distance(odds[i])  # 模拟每匹马前进的随机距离

        # bubble sort race_pos based on total distance travelled
        # in the original implementation, race_pos is reset for each
        # simulation step, so we keep this behaviour here
        race_pos = list(range(num_horses))  # 重置赛马位置列表
        for line in range(num_horses):
            for i in range(num_horses - 1 - line):
                if total_distance[race_pos[i]] < total_distance[race_pos[i + 1]]:
                    continue
                race_pos[i], race_pos[i + 1] = race_pos[i + 1], race_pos[i]  # 根据每匹马的总距离排序赛马位置列表

        # print current state of the race
        print_race_state(total_distance, race_pos)  # 打印当前赛马状态

        # goal line is defined as 28 units from start
        # check if the winning horse is already over the finish line
        if total_distance[race_pos[-1]] >= 28:  # 检查是否有马已经到达终点线
            return race_pos  # 返回赛马位置列表

        # this was not in the original BASIC implementation, but it makes the
        # race visualization a nice animation (if the terminal size is set to 31 rows)
        time.sleep(1)  # 模拟赛马过程的时间间隔


def print_race_results(race_positions, odds, bets, player_names) -> None:
    """Print the race results, as well as the winnings of each player"""

    # print the race positions first
    # 打印赛果标题
    basic_print("THE RACE RESULTS ARE:")
    # 遍历倒序的赛马位置列表，同时记录位置
    for position, horse_idx in enumerate(reversed(race_positions), start=1):
        # 根据位置和赛马编号生成一行文字
        line = f"{position} PLACE HORSE NO. {horse_idx} AT {odds[horse_idx]}:1"
        # 打印空行
        basic_print("")
        # 打印生成的文字行
        basic_print(line)
    
    # 打印玩家赢得的金额
    # 获取获胜的赛马编号
    winning_horse_idx = race_positions[-1]
    # 遍历玩家姓名列表
    for idx, name in enumerate(player_names):
        # 获取玩家下注的赛马和金额
        (horse, amount) = bets[idx]
        # 如果玩家下注的赛马和获胜的赛马编号相同
        if horse == winning_horse_idx:
            # 打印空行
            basic_print("")
            # 打印玩家赢得的金额
            basic_print(f"{name} WINS ${amount * odds[winning_horse_idx]}")
def main_loop(player_names, horse_odds) -> None:
    """Main game loop"""

    # 循环进行游戏
    while True:
        # 打印马匹赔率
        print_horse_odds(horse_odds)
        # 获取玩家的赌注
        bets = get_bets(player_names)
        # 模拟比赛，得到最终的比赛结果
        final_race_positions = simulate_race(horse_odds)
        # 打印比赛结果
        print_race_results(final_race_positions, horse_odds, bets, player_names)

        # 询问玩家是否想要在下一场比赛下注
        basic_print("DO YOU WANT TO BET ON THE NEXT RACE ?")
        one_more = basic_input("YES OR NO")
        # 如果玩家不想再下注，则跳出循环
        if one_more.upper() != "YES":
            break


def main() -> None:
    # 游戏介绍，玩家姓名和马匹赔率只生成一次
    introduction()
    player_names = setup_players()
    horse_odds = setup_horses()

    # 游戏的主循环，玩家可以进行多次比赛，赔率不变
    main_loop(player_names, horse_odds)


if __name__ == "__main__":
    main()
```