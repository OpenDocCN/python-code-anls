# `basic-computer-games\38_Fur_Trader\python\furtrader.py`

```

#!/usr/bin/env python3
# 指定脚本解释器为Python3

import random  # for generating random numbers
import sys  # for system function, like exit()
from typing import List
# 导入random模块用于生成随机数，导入sys模块用于系统函数，导入typing模块用于类型提示

# global variables for storing player's status
player_funds: float = 0  # no money
player_furs = [0, 0, 0, 0]  # no furs
# 定义全局变量用于存储玩家的状态，包括玩家资金和毛皮数量

# Constants
FUR_MINK = 0
FUR_BEAVER = 1
FUR_ERMINE = 2
FUR_FOX = 3
MAX_FURS = 190
FUR_NAMES = ["MINK", "BEAVER", "ERMINE", "FOX"]
# 定义常量，包括毛皮类型、最大毛皮数量和毛皮名称列表

FORT_MONTREAL = 1
FORT_QUEBEC = 2
FORT_NEWYORK = 3
FORT_NAMES = ["HOCHELAGA (MONTREAL)", "STADACONA (QUEBEC)", "NEW YORK"]
# 定义常量，包括要选择的砦的编号和名称列表

def show_introduction() -> None:
    """Show the player the introductory message"""
    # 显示玩家介绍信息
    print("YOU ARE THE LEADER OF A FRENCH FUR TRADING EXPEDITION IN ")
    print("1776 LEAVING THE LAKE ONTARIO AREA TO SELL FURS AND GET")
    print("SUPPLIES FOR THE NEXT YEAR.  YOU HAVE A CHOICE OF THREE")
    print("FORTS AT WHICH YOU MAY TRADE.  THE COST OF SUPPLIES")
    print("AND THE AMOUNT YOU RECEIVE FOR YOUR FURS WILL DEPEND")
    print("ON THE FORT THAT YOU CHOOSE.")
    print()

def get_fort_choice() -> int:
    """Show the player the choices of Fort, get their input, if the
    input is a valid choice (1,2,3) return it, otherwise keep
    prompting the user."""
    # 显示玩家选择砦的选项，获取玩家输入，如果输入是有效选择（1,2,3），则返回，否则继续提示用户
    result = 0
    while result == 0:
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

        player_choice = input(">> ")  # get input from the player
        # 获取玩家输入

        # try to convert the player's string input into an integer
        try:
            result = int(player_choice)  # string to integer
        except Exception:
            # Whatever the player typed, it could not be interpreted as a number
            pass

    return result

# 其余函数的注释请参照示例中的注释风格进行添加

```