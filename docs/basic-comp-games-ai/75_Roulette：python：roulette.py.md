# `d:/src/tocomm/basic-computer-games\75_Roulette\python\roulette.py`

```
import random  # 导入 random 模块，用于生成随机数
from datetime import date  # 从 datetime 模块中导入 date 类
from typing import List, Tuple  # 从 typing 模块中导入 List 和 Tuple 类型

global RED_NUMBERS  # 声明 RED_NUMBERS 为全局变量
RED_NUMBERS = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]  # 初始化 RED_NUMBERS 列表

def print_instructions() -> None:  # 定义函数 print_instructions，返回类型为 None
    print(  # 打印下面的字符串
        """
THIS IS THE BETTING LAYOUT
  (*=RED)

 1*    2     3*  # 打印赌注布局
 4     5*    6
 7*    8     9*
10    11    12*
---------------
13    14*   15
16*   # 使用字节流里面内容创建 ZIP 对象
17    bio = BytesIO(open(fname, 'rb').read())
18*   # 创建一个 ZIP 对象，用于操作 ZIP 文件
19*   zip = zipfile.ZipFile(bio, 'r')
20    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
21*   fdict = {n:zip.read(n) for n in zip.namelist()}
22    # 关闭 ZIP 对象
23*   zip.close()
24    # 返回结果字典
25*   return fdict
26    # 结束函数定义
27*   
28    # 结束函数定义
29    # 结束函数定义
30*   
31    # 结束函数定义
32*   
33    # 结束函数定义
34*   
35    # 结束函数定义
36*   
    # 结束函数定义
    # 结束函数定义
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制数据，并封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
def query_bets() -> Tuple[List[int], List[int]]:
    """Queries the user to input their bets"""
    bet_count = -1  # 初始化赌注数量为-1
    while bet_count <= 0:  # 当赌注数量小于等于0时循环
        try:
            bet_count = int(input("HOW MANY BETS? "))  # 询问用户输入赌注数量
        except Exception:  # 捕获可能的异常
            ...

    bet_ids = [-1] * bet_count  # 创建一个长度为赌注数量的列表，初始值为-1
    bet_values = [0] * bet_count  # 创建一个长度为赌注数量的列表，初始值为0

    for i in range(bet_count):  # 遍历赌注数量次
        while bet_ids[i] == -1:  # 当赌注ID为-1时循环
            try:
                in_string = input("NUMBER " + str(i + 1) + "? ").split(",")  # 询问用户输入赌注号码和值，并以逗号分隔
                id_, val = int(in_string[0]), int(in_string[1])  # 将输入的赌注号码和值转换为整数

```
```python
                # 检查其他的赌注ID
                for j in range(i):
                    if id_ != -1 and bet_ids[j] == id_:
                        id_ = -1
                        print("YOU ALREADY MADE THAT BET ONCE, DUM-DUM")
                        break

                if id_ > 0 and id_ <= 50 and val >= 5 and val <= 500:
                    bet_ids[i] = id_
                    bet_values[i] = val
            except Exception:
                pass
    return bet_ids, bet_values


def bet_results(bet_ids: List[int], bet_values: List[int], result) -> int:
    """计算结果，打印它们，并返回总净赢利"""
    total_winnings = 0

    def get_modifier(id_: int, num: int) -> int:
        # 检查条件，根据不同的条件返回不同的值
        if (
            (id_ == 37 and num <= 12)  # 如果 id_ 等于 37 并且 num 小于等于 12
            or (id_ == 38 and num > 12 and num <= 24)  # 或者 id_ 等于 38 并且 num 大于 12 并且小于等于 24
            or (id_ == 39 and num > 24 and num < 37)  # 或者 id_ 等于 39 并且 num 大于 24 并且小于 37
            or (id_ == 40 and num < 37 and num % 3 == 1)  # 或者 id_ 等于 40 并且 num 小于 37 并且 num 除以 3 的余数等于 1
            or (id_ == 41 and num < 37 and num % 3 == 2)  # 或者 id_ 等于 41 并且 num 小于 37 并且 num 除以 3 的余数等于 2
            or (id_ == 42 and num < 37 and num % 3 == 0)  # 或者 id_ 等于 42 并且 num 小于 37 并且 num 除以 3 的余数等于 0
        ):
            return 2  # 返回值为 2
        elif (
            (id_ == 43 and num <= 18)  # 如果 id_ 等于 43 并且 num 小于等于 18
            or (id_ == 44 and num > 18 and num <= 36)  # 或者 id_ 等于 44 并且 num 大于 18 并且小于等于 36
            or (id_ == 45 and num % 2 == 0)  # 或者 id_ 等于 45 并且 num 为偶数
            or (id_ == 46 and num % 2 == 1)  # 或者 id_ 等于 46 并且 num 为奇数
            or (id_ == 47 and num in RED_NUMBERS)  # 或者 id_ 等于 47 并且 num 在 RED_NUMBERS 中
            or (id_ == 48 and num not in RED_NUMBERS)  # 或者 id_ 等于 48 并且 num 不在 RED_NUMBERS 中
        ):
            return 1  # 返回值为 1
        elif id_ < 37 and id_ == num:  # 如果 id_ 小于 37 并且 id_ 等于 num
            return 35  # 返回值为 35
        else:
            return -1  # 如果条件不满足，返回-1

    for i in range(len(bet_ids)):
        winnings = bet_values[i] * get_modifier(bet_ids[i], result)  # 根据赌注和结果计算赢得的金额
        total_winnings += winnings  # 累加总赢得的金额

        if winnings >= 0:
            print("YOU WIN " + str(winnings) + " DOLLARS ON BET " + str(i + 1))  # 如果赢得的金额大于等于0，打印赢得的金额和赌注编号
        else:
            print("YOU LOSE " + str(winnings * -1) + " DOLLARS ON BET " + str(i + 1))  # 如果赢得的金额小于0，打印损失的金额和赌注编号

    return winnings  # 返回赢得的金额


def print_check(amount: int) -> None:
    """Print a check of a given amount"""
    name = input("TO WHOM SHALL I MAKE THE CHECK? ")  # 获取收款人姓名

    print("-" * 72)  # 打印分隔线
    # 打印空行
    print()
    # 打印空格和随机生成的支票号码
    print(" " * 40 + "CHECK NO. " + str(random.randint(0, 100)))
    # 打印空格和当天日期
    print(" " * 40 + str(date.today()))
    # 打印空行
    print()
    # 打印支票收款人和金额
    print("PAY TO THE ORDER OF -----" + name + "----- $" + str(amount))
    # 打印空行
    print()
    # 打印空格和银行名称
    print(" " * 40 + "THE MEMORY BANK OF NEW YORK")
    # 打印空格和计算机名称
    print(" " * 40 + "THE COMPUTER")
    # 打印空格和分隔线
    print(" " * 40 + "----------X-----")
    # 打印分隔线
    print("-" * 72)


def main() -> None:
    # 初始化玩家余额
    player_balance = 1000
    # 初始化主机余额
    host_balance = 100000

    # 打印轮盘赌游戏标题
    print(" " * 32 + "ROULETTE")
    # 打印创意计算公司和地址
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    # 打印空行
    print()
    # 打印空行
    print()
    print()  # 打印空行

    if string_to_bool(input("DO YOU WANT INSTRUCTIONS? ")):  # 如果用户输入的字符串可以转换为布尔值，且为True，则执行下面的代码
        print_instructions()  # 打印游戏说明

    while True:  # 无限循环
        bet_ids, bet_values = query_bets()  # 调用函数query_bets()获取赌注的id和值

        print("SPINNING")  # 打印"SPINNING"
        print()  # 打印空行
        print()  # 再次打印空行

        val = random.randint(0, 38)  # 生成一个0到38之间的随机整数
        if val == 38:  # 如果随机数等于38
            print("0")  # 打印"0"
        elif val == 37:  # 如果随机数等于37
            print("00")  # 打印"00"
        elif val in RED_NUMBERS:  # 如果随机数在RED_NUMBERS列表中
            print(str(val) + " RED")  # 打印随机数和"RED"
        else:  # 如果以上条件都不满足
        # 打印玩家的赢钱情况
        print(str(val) + " BLACK")

        # 计算玩家的总赢钱数，并更新玩家和主持人的余额
        total_winnings = bet_results(bet_ids, bet_values, val)
        player_balance += total_winnings
        host_balance -= total_winnings

        # 打印玩家和主持人的总余额
        print("TOTALS:\tME\t\tYOU")
        print("\t\t" + str(host_balance) + "\t" + str(player_balance)

        # 如果玩家余额小于等于0，则打印提示信息并结束游戏
        if player_balance <= 0:
            print("OOPS! YOU JUST SPENT YOUR LAST DOLLAR!")
            break
        # 如果主持人余额小于等于0，则打印提示信息并重置玩家余额为101000，并结束游戏
        elif host_balance <= 0:
            print("YOU BROKE THE HOUSE!")
            player_balance = 101000
            break
        # 如果玩家选择不再玩，则结束游戏
        if not string_to_bool(input("PLAY AGAIN? ")):
            break
    if player_balance <= 0:  # 如果玩家余额小于等于0
        print("THANKS FOR YOUR MONEY")  # 打印感谢玩家的钱
        print("I'LL USE IT TO BUY A SOLID GOLD ROULETTE WHEEL")  # 打印将用钱购买实心金轮盘
    else:  # 否则
        print_check(player_balance)  # 调用print_check函数打印玩家余额
    print("COME BACK SOON!")  # 打印请尽快回来


def string_to_bool(string: str) -> bool:
    """Converts a string to a bool"""  # 将字符串转换为布尔值的函数
    return string.lower() in ("yes", "y", "true", "t", "yes")  # 返回字符串是否在指定的布尔值列表中


if __name__ == "__main__":  # 如果当前文件被作为主程序运行
    main()  # 调用main函数
```