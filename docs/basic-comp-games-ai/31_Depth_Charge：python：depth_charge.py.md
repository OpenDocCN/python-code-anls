# `d:/src/tocomm/basic-computer-games\31_Depth_Charge\python\depth_charge.py`

```
"""
Original BASIC version as published in Basic Computer Games (1978)
https://www.atariarchives.org/basicgames/showpage.php?page=55

Converted to Python by Anson VanDoren in 2021
"""

import math  # 导入 math 模块
import random  # 导入 random 模块
from typing import Tuple  # 从 typing 模块导入 Tuple 类型


def show_welcome() -> None:  # 定义函数 show_welcome，返回类型为 None
    # Clear screen. chr(27) is `Esc`, and the control sequence is
    # initiated by Ctrl+[ 
    # `J` is "Erase in Display" and `2J` means clear the entire screen
    print(chr(27) + "[2J")  # 打印控制字符，清空屏幕

    # Show the intro text, centered
    print("DEPTH CHARGE".center(45))  # 打印标题，居中显示
    print("Creative Computing  Morristown, New Jersey\n\n".center(45))
    # 打印标题信息，居中显示在45个字符的宽度内


def get_num_charges() -> Tuple[int, int]:
    print("Depth Charge game\n")
    # 打印游戏标题信息

    while True:
        search_area_str = input("Dimensions of search area? ")
        # 获取用户输入的搜索区域尺寸

        # 确保输入为整数
        try:
            search_area = int(search_area_str)
            break
        except ValueError:
            print("Must enter an integer number. Please try again...")
        # 捕获输入不是整数的情况，提示用户重新输入

    num_charges = int(math.log2(search_area)) + 1
    # 根据搜索区域尺寸计算出需要的深度炸弹数量
    return search_area, num_charges
    # 返回搜索区域尺寸和深度炸弹数量的元组


def ask_for_new_game() -> None:
    # 请求开始新游戏
    answer = input("Another game (Y or N): ")  # 询问用户是否要再玩一次游戏，并将用户输入的值存储在answer变量中
    if answer.lower().strip()[0] == "y":  # 将用户输入的值转换为小写并去除首尾空格，然后判断第一个字符是否为'y'
        main()  # 如果用户输入的是'y'，则调用main函数
    else:  # 如果用户输入的不是'y'
        print("OK. Hope you enjoyed yourself")  # 打印消息
        exit()  # 退出程序


def show_shot_result(shot, location) -> None:  # 定义一个函数，接受两个参数shot和location，并且没有返回值
    result = "Sonar reports shot was "  # 初始化result变量
    if shot[1] > location[1]:  # 如果shot的y坐标大于location的y坐标
        result += "north"  # 在result后面添加'north'
    elif shot[1] < location[1]:  # 如果shot的y坐标小于location的y坐标
        result += "south"  # 在result后面添加'south'
    if shot[0] > location[0]:  # 如果shot的x坐标大于location的x坐标
        result += "east"  # 在result后面添加'east'
    elif shot[0] < location[0]:  # 如果shot的x坐标小于location的x坐标
        result += "west"  # 在result后面添加'west'
    if shot[1] != location[1] or shot[0] != location[0]:  # 如果shot的y坐标不等于location的y坐标或者shot的x坐标不等于location的x坐标
        result += " and "  # 在result后面添加' and '
    if shot[2] > location[2]:  # 如果射击的深度大于目标深度
        result += "too low."  # 结果字符串添加"too low."
    elif shot[2] < location[2]:  # 如果射击的深度小于目标深度
        result += "too high."  # 结果字符串添加"too high."
    else:  # 否则
        result += "depth OK."  # 结果字符串添加"depth OK."
    print(result)  # 打印结果字符串
    return  # 返回结果

def get_shot_input() -> Tuple[int, int, int]:  # 定义函数get_shot_input，返回类型为元组
    while True:  # 无限循环
        raw_guess = input("Enter coordinates: ")  # 获取用户输入的坐标
        try:  # 尝试执行以下代码
            xyz = raw_guess.split()  # 将用户输入的坐标字符串分割成列表
        except ValueError:  # 如果出现值错误
            print("Please enter coordinates separated by spaces")  # 打印提示信息
            print("Example: 3 2 1")  # 打印示例
            continue  # 继续下一次循环
        try:
            x, y, z = (int(num) for num in xyz)  # 尝试将xyz中的每个元素转换为整数并赋值给x, y, z
            return x, y, z  # 如果转换成功，则返回x, y, z
        except ValueError:  # 如果转换失败
            print("Please enter whole numbers only")  # 打印错误提示信息

def play_game(search_area, num_charges) -> None:
    print("\nYou are the captain of the destroyer USS Computer.")  # 打印游戏开始的提示信息
    print("An enemy sub has been causing you trouble. Your")  # 打印游戏背景信息
    print(f"mission is to destroy it. You have {num_charges} shots.")  # 打印玩家拥有的发射次数
    print("Specify depth charge explosion point with a")  # 打印提示信息
    print("trio of numbers -- the first two are the")  # 打印提示信息
    print("surface coordinates; the third is the depth.")  # 打印提示信息
    print("\nGood luck!\n")  # 打印祝福信息

    # Generate position for submarine
    a, b, c = (random.randint(0, search_area) for _ in range(3))  # 生成潜艇的位置坐标

    # Get inputs until win or lose
    for i in range(num_charges):  # 循环执行 num_charges 次
        print(f"\nTrial #{i+1}")  # 打印当前尝试的次数
        x, y, z = get_shot_input()  # 获取用户输入的射击坐标

        if (x, y, z) == (a, b, c):  # 判断用户输入的坐标是否与目标坐标相同
            print(f"\nB O O M ! ! You found it in {i+1} tries!\n")  # 如果相同则打印找到目标的消息
            ask_for_new_game()  # 请求开始新游戏
        else:
            show_shot_result((x, y, z), (a, b, c))  # 显示射击结果

    # out of shots
    print("\nYou have been torpedoed! Abandon ship!")  # 打印用户用尽所有尝试机会的消息
    print(f"The submarine was at {a} {b} {c}")  # 打印潜艇的位置坐标
    ask_for_new_game()  # 请求开始新游戏


def main() -> None:
    search_area, num_charges = get_num_charges()  # 获取搜索区域和尝试次数
    play_game(search_area, num_charges)  # 调用 play_game 函数开始游戏
# 如果当前脚本被直接执行，则调用 main() 函数
# 这个条件语句检查当前脚本是否被直接执行，如果是，则调用 main() 函数。这通常用于将脚本作为可执行文件运行时执行特定的操作。
```