# `d:/src/tocomm/basic-computer-games\38_Fur_Trader\python\furtrader.py`

```
#!/usr/bin/env python3  # 指定脚本解释器为 Python3

import random  # 导入 random 模块，用于生成随机数
import sys  # 导入 sys 模块，用于系统函数，如 exit()
from typing import List  # 导入 typing 模块中的 List 类型

# 全局变量，用于存储玩家的状态
player_funds: float = 0  # 玩家没有钱
player_furs = [0, 0, 0, 0]  # 玩家没有毛皮


# 常量
FUR_MINK = 0  # 毛皮类型为貂
FUR_BEAVER = 1  # 毛皮类型为海狸
FUR_ERMINE = 2  # 毛皮类型为鼬
FUR_FOX = 3  # 毛皮类型为狐狸
MAX_FURS = 190  # 最大毛皮数量为 190
FUR_NAMES = ["MINK", "BEAVER", "ERMINE", "FOX"]  # 毛皮类型名称列表
FORT_MONTREAL = 1  # 蒙特利尔要塞编号为 1
    # 显示玩家介绍信息
    print("YOU ARE THE LEADER OF A FRENCH FUR TRADING EXPEDITION IN ")
    print("1776 LEAVING THE LAKE ONTARIO AREA TO SELL FURS AND GET")
    print("SUPPLIES FOR THE NEXT YEAR.  YOU HAVE A CHOICE OF THREE")
    print("FORTS AT WHICH YOU MAY TRADE.  THE COST OF SUPPLIES")
    print("AND THE AMOUNT YOU RECEIVE FOR YOUR FURS WILL DEPEND")
    print("ON THE FORT THAT YOU CHOOSE.")
    print()

    # 获取玩家选择的要前往的堡垒
    """显示玩家堡垒的选择，获取他们的输入，如果输入是有效的选择（1,2,3），则返回它，否则继续提示用户。"""
    result = 0  # 初始化变量result为0
    while result == 0:  # 当result为0时执行循环
        print()  # 打印空行
        print("YOU MAY TRADE YOUR FURS AT FORT 1, FORT 2,")  # 打印提示信息
        print("OR FORT 3.  FORT 1 IS FORT HOCHELAGA (MONTREAL)")  # 打印提示信息
        print("AND IS UNDER THE PROTECTION OF THE FRENCH ARMY.")  # 打印提示信息
        print("FORT 2 IS FORT STADACONA (QUEBEC) AND IS UNDER THE")  # 打印提示信息
        print("PROTECTION OF THE FRENCH ARMY.  HOWEVER, YOU MUST")  # 打印提示信息
        print("MAKE A PORTAGE AND CROSS THE LACHINE RAPIDS.")  # 打印提示信息
        print("FORT 3 IS FORT NEW YORK AND IS UNDER DUTCH CONTROL.")  # 打印提示信息
        print("YOU MUST CROSS THROUGH IROQUOIS LAND.")  # 打印提示信息
        print("ANSWER 1, 2, OR 3.")  # 打印提示信息

        player_choice = input(">> ")  # 从玩家获取输入

        # 尝试将玩家的字符串输入转换为整数
        try:
            result = int(player_choice)  # 将字符串转换为整数
        except Exception:
            # 无论玩家输入了什么，都无法解释为数字
            pass
```
这是一个占位符，表示在这个位置没有需要执行的代码。

```
    return result
```
返回result变量的值。

```
def show_fort_comment(which_fort) -> None:
```
定义一个名为show_fort_comment的函数，它接受一个名为which_fort的参数，并且不返回任何值。

```
    """Print the description for the fort"""
```
这是函数的文档字符串，用于描述函数的作用。

```
    print()
```
打印一个空行。

```
    if which_fort == FORT_MONTREAL:
```
如果which_fort等于FORT_MONTREAL，则执行以下代码。

```
        print("YOU HAVE CHOSEN THE EASIEST ROUTE.  HOWEVER, THE FORT")
        print("IS FAR FROM ANY SEAPORT.  THE VALUE")
        print("YOU RECEIVE FOR YOUR FURS WILL BE LOW AND THE COST")
        print("OF SUPPLIES HIGHER THAN AT FORTS STADACONA OR NEW YORK.")
```
打印一系列描述FORT_MONTREAL的文本。

```
    elif which_fort == FORT_QUEBEC:
```
如果which_fort等于FORT_QUEBEC，则执行以下代码。

```
        print("YOU HAVE CHOSEN A HARD ROUTE.  IT IS, IN COMPARSION,")
        print("HARDER THAN THE ROUTE TO HOCHELAGA BUT EASIER THAN")
        print("THE ROUTE TO NEW YORK.  YOU WILL RECEIVE AN AVERAGE VALUE")
        print("FOR YOUR FURS AND THE COST OF YOUR SUPPLIES WILL BE AVERAGE.")
```
打印一系列描述FORT_QUEBEC的文本。

```
    elif which_fort == FORT_NEWYORK:
```
如果which_fort等于FORT_NEWYORK，则执行以下代码。

```
        print("YOU HAVE CHOSEN THE MOST DIFFICULT ROUTE.  AT")
```
打印一系列描述FORT_NEWYORK的文本。
        print("FORT NEW YORK YOU WILL RECEIVE THE HIGHEST VALUE")  # 打印消息
        print("FOR YOUR FURS.  THE COST OF YOUR SUPPLIES")  # 打印消息
        print("WILL BE LOWER THAN AT ALL THE OTHER FORTS.")  # 打印消息
    else:
        print("Internal error #1, fort " + str(which_fort) + " does not exist")  # 如果条件不满足，打印错误消息
        sys.exit(1)  # 退出程序，返回错误码1
    print()  # 打印空行


def get_yes_or_no() -> str:
    """Prompt the player to enter 'YES' or 'NO'. Keep prompting until
    valid input is entered.  Accept various spellings by only
    checking the first letter of input.
    Return a single letter 'Y' or 'N'"""
    result = ""  # 初始化结果变量
    while result not in ("Y", "N"):  # 当结果不是Y或N时循环
        print("ANSWER YES OR NO")  # 打印提示消息
        player_choice = input(">> ")  # 获取玩家输入
        player_choice = player_choice.strip().upper()  # 去除空格并转换为大写
        if player_choice.startswith("Y"):  # 如果玩家输入以Y开头
            result = "Y"  # 如果玩家选择以 "Y" 开头的选项，将结果设置为 "Y"
        elif player_choice.startswith("N"):  # 如果玩家选择以 "N" 开头的选项，将结果设置为 "N"
            result = "N"
    return result  # 返回结果


def get_furs_purchase() -> List[int]:
    """Prompt the player for how many of each fur type they want.
    Accept numeric inputs, re-prompting on incorrect input values"""
    results: List[int] = []  # 创建一个空列表用于存储玩家输入的皮毛数量

    print("YOUR " + str(MAX_FURS) + " FURS ARE DISTRIBUTED AMONG THE FOLLOWING")
    print("KINDS OF PELTS: MINK, BEAVER, ERMINE AND FOX.")
    print()

    while len(results) < len(FUR_NAMES):  # 当结果列表的长度小于皮毛名称列表的长度时执行循环
        print(f"HOW MANY {FUR_NAMES[len(results)]} DO YOU HAVE")  # 提示玩家输入特定类型的皮毛数量
        count_str = input(">> ")  # 获取玩家输入的数量
        try:
            count = int(count_str)  # 尝试将输入的数量转换为整数
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
            player_funds: float = 600  # Initial player start money  # 初始化玩家的起始资金
            player_furs = [0, 0, 0, 0]  # Player fur inventory  # 玩家的皮毛库存

            print("DO YOU WISH TO TRADE FURS?")  # 打印询问玩家是否愿意交易皮毛
            should_trade = get_yes_or_no()  # 获取玩家的交易选择
            if should_trade == "N":  # 如果玩家选择不交易
                sys.exit(0)  # STOP  # 程序退出
            game_state = "trading"  # 设置游戏状态为交易状态

        elif game_state == "trading":  # 如果游戏状态为交易状态
            print()
            print("YOU HAVE $ %1.2f IN SAVINGS" % (player_funds))  # 打印玩家的存款金额
            print("AND " + str(MAX_FURS) + " FURS TO BEGIN THE EXPEDITION")  # 打印玩家开始远征时的皮毛数量
            player_furs = get_furs_purchase()  # 获取玩家购买的皮毛数量

            if sum(player_furs) > MAX_FURS:  # 如果玩家购买的皮毛数量超过最大值
                print()
                print("YOU MAY NOT HAVE THAT MANY FURS.")  # 打印提示玩家不可购买那么多皮毛
                print("DO NOT TRY TO CHEAT.  I CAN ADD.")  # 打印提示不要作弊
                print("YOU MUST START AGAIN.")  # 打印提示玩家必须重新开始
                game_state = "starting"  # 设置游戏状态为“开始”，如果条件不满足则设置为“选择要塞”
            else:
                game_state = "choosing fort"

        elif game_state == "choosing fort":  # 如果游戏状态为“选择要塞”
            which_fort = get_fort_choice()  # 获取要塞选择
            show_fort_comment(which_fort)  # 显示要塞评论
            print("DO YOU WANT TO TRADE AT ANOTHER FORT?")  # 打印提示信息
            change_fort = get_yes_or_no()  # 获取是否要更换要塞的选择
            if change_fort == "N":  # 如果选择不更换要塞
                game_state = "travelling"  # 设置游戏状态为“旅行”

        elif game_state == "travelling":  # 如果游戏状态为“旅行”
            print()  # 打印空行
            if which_fort == FORT_MONTREAL:  # 如果选择的要塞是蒙特利尔
                mink_price = (  # 计算貂皮价格
                    int((0.2 * random.random() + 0.70) * 100 + 0.5) / 100
                )  # 使用随机数计算价格
                ermine_price = (  # 计算貂皮价格
                    int((0.2 * random.random() + 0.65) * 100 + 0.5) / 100
                )  # 计算海狸价格，根据随机数生成价格并四舍五入到小数点后两位
                beaver_price = (
                    int((0.2 * random.random() + 0.75) * 100 + 0.5) / 100
                )  # 计算海狸价格，根据随机数生成价格并四舍五入到小数点后两位
                fox_price = (
                    int((0.2 * random.random() + 0.80) * 100 + 0.5) / 100
                )  # 计算狐狸价格，根据随机数生成价格并四舍五入到小数点后两位

                print("SUPPLIES AT FORT HOCHELAGA COST $150.00.")
                print("YOUR TRAVEL EXPENSES TO HOCHELAGA WERE $10.00.")
                player_funds -= 160

            elif which_fort == FORT_QUEBEC:
                mink_price = (
                    int((0.30 * random.random() + 0.85) * 100 + 0.5) / 100
                )  # 计算貂皮价格，根据随机数生成价格并四舍五入到小数点后两位
                ermine_price = (
                    int((0.15 * random.random() + 0.80) * 100 + 0.5) / 100
                )  # 计算貂皮价格，根据随机数生成价格并四舍五入到小数点后两位
                beaver_price = (
                    int((0.20 * random.random() + 0.90) * 100 + 0.5) / 100
                )  # Calculate the price of beaver pelts using a random factor
                fox_price = (
                    int((0.25 * random.random() + 1.10) * 100 + 0.5) / 100
                )  # Calculate the price of fox pelts using a random factor
                event_picker = int(10 * random.random()) + 1  # Randomly pick an event

                if event_picker <= 2:  # If the event picker is less than or equal to 2
                    print("YOUR BEAVER WERE TOO HEAVY TO CARRY ACROSS")
                    print("THE PORTAGE.  YOU HAD TO LEAVE THE PELTS, BUT FOUND")
                    print("THEM STOLEN WHEN YOU RETURNED.")
                    player_furs[FUR_BEAVER] = 0  # Set the number of beaver pelts to 0
                elif event_picker <= 6:  # If the event picker is less than or equal to 6
                    print("YOU ARRIVED SAFELY AT FORT STADACONA.")
                elif event_picker <= 8:  # If the event picker is less than or equal to 8
                    print("YOUR CANOE UPSET IN THE LACHINE RAPIDS.  YOU")
                    print("LOST ALL YOUR FURS.")
                    player_furs = [0, 0, 0, 0]  # Set all fur counts to 0
                elif event_picker <= 10:  # If the event picker is less than or equal to 10
                    print("YOUR FOX PELTS WERE NOT CURED PROPERLY.")
                    print("NO ONE WILL BUY THEM.")  # 打印消息，表示没有人愿意购买
                    player_furs[FUR_FOX] = 0  # 将玩家的狐狸皮数量设为0
                else:
                    print(
                        "Internal Error #3, Out-of-bounds event_picker"
                        + str(event_picker)
                    )  # 打印错误消息，表示内部错误 #3，超出范围的事件选择器
                    sys.exit(1)  # 退出程序，表示有一个bug

                print()  # 打印空行
                print("SUPPLIES AT FORT STADACONA COST $125.00.")  # 打印消息，表示在Stadacona堡垒的供应品价格为$125.00
                print("YOUR TRAVEL EXPENSES TO STADACONA WERE $15.00.")  # 打印消息，表示你前往Stadacona的旅行费用为$15.00
                player_funds -= 140  # 玩家的资金减去140

            elif which_fort == FORT_NEWYORK:
                mink_price = (
                    int((0.15 * random.random() + 1.05) * 100 + 0.5) / 100
                )  # 计算貂皮的价格
                ermine_price = (
                    int((0.15 * random.random() + 0.95) * 100 + 0.5) / 100
                )  # 计算貂皮的价格
                )  # 生成一个随机的浮点数，表示海狸的价格
                beaver_price = (
                    int((0.25 * random.random() + 1.00) * 100 + 0.5) / 100
                )  # 生成一个随机的浮点数，表示海狸的价格
                if fox_price is None:
                    # 原始 Bug？纽约没有生成狐狸的价格，将使用之前的“D1”价格
                    # 如果之前没有数值，就随机生成一个
                    fox_price = (
                        int((0.25 * random.random() + 1.05) * 100 + 0.5) / 100
                    )  # 不在原始代码中
                event_picker = int(10 * random.random()) + 1

                if event_picker <= 2:
                    print("YOU WERE ATTACKED BY A PARTY OF IROQUOIS.")
                    print("ALL PEOPLE IN YOUR TRADING GROUP WERE")
                    print("KILLED.  THIS ENDS THE GAME.")
                    sys.exit(0)
                elif event_picker <= 6:
                    print("YOU WERE LUCKY.  YOU ARRIVED SAFELY")
                    print("AT FORT NEW YORK.")
                elif event_picker <= 8:  # 如果事件选择器小于等于8
                    print("YOU NARROWLY ESCAPED AN IROQUOIS RAIDING PARTY.")  # 打印玩家逃脱伊罗quois袭击队的消息
                    print("HOWEVER, YOU HAD TO LEAVE ALL YOUR FURS BEHIND.")  # 打印玩家不得不留下所有毛皮的消息
                    player_furs = [0, 0, 0, 0]  # 重置玩家的毛皮数量为0
                elif event_picker <= 10:  # 如果事件选择器小于等于10
                    mink_price /= 2  # 貂皮价格减半
                    fox_price /= 2  # 狐狸皮价格减半
                    print("YOUR MINK AND BEAVER WERE DAMAGED ON YOUR TRIP.")  # 打印玩家的貂皮和海狸皮在旅途中受损的消息
                    print("YOU RECEIVE ONLY HALF THE CURRENT PRICE FOR THESE FURS.")  # 打印玩家只能得到这些毛皮当前价格的一半的消息
                else:  # 如果事件选择器大于10
                    print(
                        "Internal Error #4, Out-of-bounds event_picker"
                        + str(event_picker)
                    )  # 打印内部错误＃4，超出范围的事件选择器的消息
                    sys.exit(1)  # 退出程序，显示有错误
                print()  # 打印空行
                print("SUPPLIES AT NEW YORK COST $85.00.")  # 打印纽约的供应品价格为$85.00
                print("YOUR TRAVEL EXPENSES TO NEW YORK WERE $25.00.")  # 打印你去纽约的旅行费用为$25.00
                player_funds -= 105  # 玩家资金减去105
            else:
                # 如果条件不满足，打印错误信息并退出程序
                print("Internal error #2, fort " + str(which_fort) + " does not exist")
                sys.exit(1)  # you have a bug

            # 计算销售额
            beaver_value = beaver_price * player_furs[FUR_BEAVER]
            fox_value = fox_price * player_furs[FUR_FOX]
            ermine_value = ermine_price * player_furs[FUR_ERMINE]
            mink_value = mink_price * player_furs[FUR_MINK]

            # 打印每种皮毛的销售额
            print()
            print("YOUR BEAVER SOLD FOR $%6.2f" % (beaver_value))
            print("YOUR FOX SOLD FOR    $%6.2f" % (fox_value))
            print("YOUR ERMINE SOLD FOR $%6.2f" % (ermine_value))
            print("YOUR MINK SOLD FOR   $%6.2f" % (mink_value))

            # 更新玩家资金
            player_funds += beaver_value + fox_value + ermine_value + mink_value

            print()
            print(
                "YOU NOW HAVE $ %1.2f INCLUDING YOUR PREVIOUS SAVINGS" % (player_funds)
            )  # 打印玩家当前的资金，包括之前的储蓄

            print()  # 打印空行
            print("DO YOU WANT TO TRADE FURS NEXT YEAR?")  # 打印询问玩家是否想要在下一年交易毛皮
            should_trade = get_yes_or_no()  # 获取玩家的输入，判断是否要交易毛皮
            if should_trade == "N":  # 如果玩家选择不交易
                sys.exit(0)  # 停止程序
            else:  # 如果玩家选择交易
                game_state = "trading"  # 修改游戏状态为交易中


if __name__ == "__main__":
    main()  # 调用主函数开始游戏
```