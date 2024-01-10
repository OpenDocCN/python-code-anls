# `basic-computer-games\38_Fur_Trader\python\furtrader.py`

```
#!/usr/bin/env python3

import random  # 用于生成随机数
import sys  # 用于系统函数，比如 exit()
from typing import List

# 存储玩家状态的全局变量
player_funds: float = 0  # 没有钱
player_furs = [0, 0, 0, 0]  # 没有毛皮


# 常量
FUR_MINK = 0
FUR_BEAVER = 1
FUR_ERMINE = 2
FUR_FOX = 3
MAX_FURS = 190
FUR_NAMES = ["MINK", "BEAVER", "ERMINE", "FOX"]
FORT_MONTREAL = 1
FORT_QUEBEC = 2
FORT_NEWYORK = 3
FORT_NAMES = ["HOCHELAGA (MONTREAL)", "STADACONA (QUEBEC)", "NEW YORK"]


def show_introduction() -> None:
    """展示玩家介绍信息"""
    print("YOU ARE THE LEADER OF A FRENCH FUR TRADING EXPEDITION IN ")
    print("1776 LEAVING THE LAKE ONTARIO AREA TO SELL FURS AND GET")
    print("SUPPLIES FOR THE NEXT YEAR.  YOU HAVE A CHOICE OF THREE")
    print("FORTS AT WHICH YOU MAY TRADE.  THE COST OF SUPPLIES")
    print("AND THE AMOUNT YOU RECEIVE FOR YOUR FURS WILL DEPEND")
    print("ON THE FORT THAT YOU CHOOSE.")
    print()


def get_fort_choice() -> int:
    """展示玩家选择要前往的堡垒，获取他们的输入，如果输入是有效选择（1,2,3），则返回它，否则继续提示用户。"""
    result = 0
    # 当结果为0时，执行以下循环
    while result == 0:
        # 打印提示信息，告诉玩家可以在三个堡垒进行皮毛交易
        print()
        print("YOU MAY TRADE YOUR FURS AT FORT 1, FORT 2,")
        print("OR FORT 3.  FORT 1 IS FORT HOCHELAGA (MONTREAL)")
        print("AND IS UNDER THE PROTECTION OF THE FRENCH ARMY.")
        print("FORT 2 IS FORT STADACONA (QUEBEC) AND IS UNDER THE")
        print("PROTECTION OF THE FRENCH ARMY.  HOWEVER, YOU MUST")
        print("MAKE A PORTAGE AND CROSS THE LACHINE RAPIDS.")
        print("FORT 3 IS FORT NEW YORK AND IS UNDER DUTCH CONTROL.")
        print("YOU MUST CROSS THROUGH IROQUOIS LAND.")
        print("ANSWER 1, 2, OR 3.")

        player_choice = input(">> ")  # 从玩家获取输入

        # 尝试将玩家的字符串输入转换为整数
        try:
            result = int(player_choice)  # 字符串转整数
        except Exception:
            # 无论玩家输入了什么，都无法解释为数字
            pass

    # 返回结果
    return result
# 打印要选择的炮台的描述
def show_fort_comment(which_fort) -> None:
    """Print the description for the fort"""
    # 打印空行
    print()
    # 根据选择的炮台不同，打印不同的描述
    if which_fort == FORT_MONTREAL:
        print("YOU HAVE CHOSEN THE EASIEST ROUTE.  HOWEVER, THE FORT")
        print("IS FAR FROM ANY SEAPORT.  THE VALUE")
        print("YOU RECEIVE FOR YOUR FURS WILL BE LOW AND THE COST")
        print("OF SUPPLIES HIGHER THAN AT FORTS STADACONA OR NEW YORK.")
    elif which_fort == FORT_QUEBEC:
        print("YOU HAVE CHOSEN A HARD ROUTE.  IT IS, IN COMPARSION,")
        print("HARDER THAN THE ROUTE TO HOCHELAGA BUT EASIER THAN")
        print("THE ROUTE TO NEW YORK.  YOU WILL RECEIVE AN AVERAGE VALUE")
        print("FOR YOUR FURS AND THE COST OF YOUR SUPPLIES WILL BE AVERAGE.")
    elif which_fort == FORT_NEWYORK:
        print("YOU HAVE CHOSEN THE MOST DIFFICULT ROUTE.  AT")
        print("FORT NEW YORK YOU WILL RECEIVE THE HIGHEST VALUE")
        print("FOR YOUR FURS.  THE COST OF YOUR SUPPLIES")
        print("WILL BE LOWER THAN AT ALL THE OTHER FORTS.")
    else:
        print("Internal error #1, fort " + str(which_fort) + " does not exist")
        sys.exit(1)  # you have a bug
    # 打印空行
    print()


# 提示玩家输入'YES'或'NO'，直到输入有效值为止。通过检查输入的第一个字母，接受各种拼写方式
# 返回单个字母'Y'或'N'
def get_yes_or_no() -> str:
    """Prompt the player to enter 'YES' or 'NO'. Keep prompting until
    valid input is entered.  Accept various spellings by only
    checking the first letter of input.
    Return a single letter 'Y' or 'N'"""
    # 初始化结果为空字符串
    result = ""
    # 当结果不是'Y'或'N'时，循环提示玩家输入
    while result not in ("Y", "N"):
        print("ANSWER YES OR NO")
        # 玩家输入选择
        player_choice = input(">> ")
        # 去除空格并转换为大写
        player_choice = player_choice.strip().upper()  # trim spaces, make upper-case
        # 如果玩家选择以'Y'开头，则结果为'Y'
        if player_choice.startswith("Y"):
            result = "Y"
        # 如果玩家选择以'N'开头，则结果为'N'
        elif player_choice.startswith("N"):
            result = "N"
    # 返回结果
    return result


# 提示玩家输入每种皮毛的数量。接受数字输入，对不正确的输入值重新提示
def get_furs_purchase() -> List[int]:
    """Prompt the player for how many of each fur type they want.
    Accept numeric inputs, re-prompting on incorrect input values"""
    # 初始化结果列表
    results: List[int] = []
    # 打印消息，显示最大毛皮数量，并列出毛皮的种类
    print("YOUR " + str(MAX_FURS) + " FURS ARE DISTRIBUTED AMONG THE FOLLOWING")
    print("KINDS OF PELTS: MINK, BEAVER, ERMINE AND FOX.")
    print()
    
    # 当结果列表长度小于毛皮种类数量时，循环询问用户拥有的每种毛皮数量
    while len(results) < len(FUR_NAMES):
        # 打印消息，询问用户拥有的某种毛皮数量
        print(f"HOW MANY {FUR_NAMES[len(results)]} DO YOU HAVE")
        # 获取用户输入的数量
        count_str = input(">> ")
        try:
            # 尝试将输入的数量转换为整数，并将其添加到结果列表中
            count = int(count_str)
            results.append(count)
        except Exception:  # 捕获异常，表示输入无效，重新循环询问
            pass
    # 返回结果列表
    return results
# 定义主函数，不返回任何数值
def main() -> None:
    # 打印游戏标题
    print(" " * 31 + "FUR TRADER")
    # 打印游戏信息
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    # 打印游戏信息
    print(" " * 15 + "(Ported to Python Oct 2012 krt@krt.com.au)")
    # 打印空行
    print("\n\n\n")

    # 初始化游戏状态为"starting"
    game_state = "starting"
    # 初始化狐狸价格为None，有时可能会取到"last"价格（可能是一个bug）
    fox_price = None  

# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```