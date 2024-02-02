# `basic-computer-games\75_Roulette\python\roulette.py`

```py
# 导入 random 模块
import random
# 从 datetime 模块中导入 date 类
from datetime import date
# 从 typing 模块中导入 List 和 Tuple 类型
from typing import List, Tuple

# 定义全局变量 RED_NUMBERS，并初始化为包含红色数字的列表
global RED_NUMBERS
RED_NUMBERS = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]

# 定义打印游戏说明的函数
def print_instructions() -> None:
    # 打印游戏规则说明
    print(
        """
        ...
        [游戏规则说明内容]
        ...
        """
    )

# 定义查询用户下注的函数
def query_bets() -> Tuple[List[int], List[int]]:
    """Queries the user to input their bets"""
    # 初始化下注数量为负数，用于循环直到输入正确的下注数量
    bet_count = -1
    # 循环直到输入正确的下注数量
    while bet_count <= 0:
        try:
            # 提示用户输入下注数量，并将输入转换为整数
            bet_count = int(input("HOW MANY BETS? "))
        except Exception:
            # 捕获异常，不做任何处理，继续循环
            ...

    # 初始化下注编号列表，长度为下注数量，初始值为-1
    bet_ids = [-1] * bet_count
    # 初始化下注金额列表，长度为下注数量，初始值为0
    bet_values = [0] * bet_count
    # 遍历下注次数的范围
    for i in range(bet_count):
        # 当下注 ID 为 -1 时，循环执行以下操作
        while bet_ids[i] == -1:
            try:
                # 从用户输入中获取下注号码和金额，并转换为整数
                in_string = input("NUMBER " + str(i + 1) + "? ").split(",")
                id_, val = int(in_string[0]), int(in_string[1])

                # 检查其他下注 ID 是否重复
                for j in range(i):
                    if id_ != -1 and bet_ids[j] == id_:
                        id_ = -1
                        print("YOU ALREADY MADE THAT BET ONCE, DUM-DUM")
                        break

                # 如果下注 ID 在 1 到 50 之间，下注金额在 5 到 500 之间，则更新下注 ID 和下注金额
                if id_ > 0 and id_ <= 50 and val >= 5 and val <= 500:
                    bet_ids[i] = id_
                    bet_values[i] = val
            # 捕获任何异常
            except Exception:
                pass
    # 返回下注 ID 和下注金额
    return bet_ids, bet_values
def bet_results(bet_ids: List[int], bet_values: List[int], result) -> int:
    """计算结果，打印它们，并返回总净赢利"""
    total_winnings = 0

    def get_modifier(id_: int, num: int) -> int:
        # 根据不同的赌注ID和结果数字计算赔率
        if (
            (id_ == 37 and num <= 12)
            or (id_ == 38 and num > 12 and num <= 24)
            or (id_ == 39 and num > 24 and num < 37)
            or (id_ == 40 and num < 37 and num % 3 == 1)
            or (id_ == 41 and num < 37 and num % 3 == 2)
            or (id_ == 42 and num < 37 and num % 3 == 0)
        ):
            return 2
        elif (
            (id_ == 43 and num <= 18)
            or (id_ == 44 and num > 18 and num <= 36)
            or (id_ == 45 and num % 2 == 0)
            or (id_ == 46 and num % 2 == 1)
            or (id_ == 47 and num in RED_NUMBERS)
            or (id_ == 48 and num not in RED_NUMBERS)
        ):
            return 1
        elif id_ < 37 and id_ == num:
            return 35
        else:
            return -1

    for i in range(len(bet_ids)):
        # 计算每个赌注的赢利
        winnings = bet_values[i] * get_modifier(bet_ids[i], result)
        total_winnings += winnings

        if winnings >= 0:
            print("YOU WIN " + str(winnings) + " DOLLARS ON BET " + str(i + 1))
        else:
            print("YOU LOSE " + str(winnings * -1) + " DOLLARS ON BET " + str(i + 1))

    return winnings


def print_check(amount: int) -> None:
    """打印指定金额的支票"""
    name = input("TO WHOM SHALL I MAKE THE CHECK? ")

    print("-" * 72)
    print()
    print(" " * 40 + "CHECK NO. " + str(random.randint(0, 100)))
    print(" " * 40 + str(date.today()))
    print()
    print("PAY TO THE ORDER OF -----" + name + "----- $" + str(amount))
    print()
    print(" " * 40 + "THE MEMORY BANK OF NEW YORK")
    print(" " * 40 + "THE COMPUTER")
    print(" " * 40 + "----------X-----")
    print("-" * 72)


def main() -> None:
    player_balance = 1000
    host_balance = 100000
    # 打印标题 "ROULETTE"
    print(" " * 32 + "ROULETTE")
    # 打印创意计算公司的地址
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    # 打印空行
    print()
    print()
    print()
    
    # 如果用户想要游戏说明，则打印游戏说明
    if string_to_bool(input("DO YOU WANT INSTRUCTIONS? ")):
        print_instructions()
    
    # 无限循环，直到用户选择退出游戏
    while True:
        # 查询用户下注的赌注编号和赌注值
        bet_ids, bet_values = query_bets()
    
        # 打印 "SPINNING"
        print("SPINNING")
        # 打印空行
        print()
        print()
    
        # 生成一个随机数作为轮盘结果
        val = random.randint(0, 38)
        # 根据轮盘结果打印相应的数字或颜色
        if val == 38:
            print("0")
        elif val == 37:
            print("00")
        elif val in RED_NUMBERS:
            print(str(val) + " RED")
        else:
            print(str(val) + " BLACK")
    
        # 计算赌注结果，并更新玩家和庄家的余额
        total_winnings = bet_results(bet_ids, bet_values, val)
        player_balance += total_winnings
        host_balance -= total_winnings
    
        # 打印玩家和庄家的余额
        print()
        print("TOTALS:\tME\t\tYOU")
        print("\t\t" + str(host_balance) + "\t" + str(player_balance)
    
        # 如果玩家余额小于等于0，则打印信息并结束游戏
        if player_balance <= 0:
            print("OOPS! YOU JUST SPENT YOUR LAST DOLLAR!")
            break
        # 如果庄家余额小于等于0，则打印信息并重置玩家余额
        elif host_balance <= 0:
            print("YOU BROKE THE HOUSE!")
            player_balance = 101000
            break
        # 如果玩家选择不再玩，则结束游戏
        if not string_to_bool(input("PLAY AGAIN? ")):
            break
    
    # 如果玩家余额小于等于0，则打印信息
    if player_balance <= 0:
        print("THANKS FOR YOUR MONEY")
        print("I'LL USE IT TO BUY A SOLID GOLD ROULETTE WHEEL")
    # 否则，打印玩家余额
    else:
        print_check(player_balance)
    # 打印结束语
    print("COME BACK SOON!")
# 将字符串转换为布尔值的函数
def string_to_bool(string: str) -> bool:
    # 将字符串转换为小写，并检查是否在指定的字符串列表中
    return string.lower() in ("yes", "y", "true", "t", "yes")

# 如果当前脚本被直接执行，则调用 main 函数
if __name__ == "__main__":
    main()
```