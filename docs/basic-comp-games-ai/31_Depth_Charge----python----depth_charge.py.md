# `basic-computer-games\31_Depth_Charge\python\depth_charge.py`

```py
"""
Original BASIC version as published in Basic Computer Games (1978)
https://www.atariarchives.org/basicgames/showpage.php?page=55

Converted to Python by Anson VanDoren in 2021
"""

import math  # 导入 math 模块
import random  # 导入 random 模块
from typing import Tuple  # 从 typing 模块导入 Tuple 类型


def show_welcome() -> None:  # 定义一个没有返回值的函数 show_welcome
    # 清空屏幕。chr(27) 是 `Esc`，控制序列由 Ctrl+[ 发起
    # `J` 是 "Erase in Display"，`2J` 表示清空整个屏幕
    print(chr(27) + "[2J")

    # 居中显示介绍文本
    print("DEPTH CHARGE".center(45))
    print("Creative Computing  Morristown, New Jersey\n\n".center(45))


def get_num_charges() -> Tuple[int, int]:  # 定义一个返回元组的函数 get_num_charges
    print("Depth Charge game\n")
    while True:
        search_area_str = input("Dimensions of search area? ")

        # 确保输入是一个整数
        try:
            search_area = int(search_area_str)
            break
        except ValueError:
            print("Must enter an integer number. Please try again...")

    num_charges = int(math.log2(search_area)) + 1
    return search_area, num_charges  # 返回搜索区域和深度炸弹数量的元组


def ask_for_new_game() -> None:  # 定义一个没有返回值的函数 ask_for_new_game
    answer = input("Another game (Y or N): ")
    if answer.lower().strip()[0] == "y":
        main()  # 调用主函数开始新游戏
    else:
        print("OK. Hope you enjoyed yourself")
        exit()  # 退出游戏


def show_shot_result(shot, location) -> None:  # 定义一个没有返回值的函数 show_shot_result，接受 shot 和 location 两个参数
    result = "Sonar reports shot was "
    if shot[1] > location[1]:  # y 方向
        result += "north"
    elif shot[1] < location[1]:  # y 方向
        result += "south"
    if shot[0] > location[0]:  # x 方向
        result += "east"
    elif shot[0] < location[0]:  # x 方向
        result += "west"
    if shot[1] != location[1] or shot[0] != location[0]:
        result += " and "

    if shot[2] > location[2]:
        result += "too low."
    elif shot[2] < location[2]:
        result += "too high."
    else:
        result += "depth OK."
    print(result)  # 打印结果
    return  # 返回


def get_shot_input() -> Tuple[int, int, int]:  # 定义一个返回元组的函数 get_shot_input
    # 无限循环，等待用户输入坐标
    while True:
        # 获取用户输入的原始猜测
        raw_guess = input("Enter coordinates: ")
        # 尝试将原始猜测按空格分割成坐标列表
        try:
            xyz = raw_guess.split()
        # 如果出现数值错误，提示用户输入正确格式的坐标
        except ValueError:
            print("Please enter coordinates separated by spaces")
            print("Example: 3 2 1")
            # 继续下一次循环
            continue
        # 尝试将坐标列表中的字符串转换为整数
        try:
            # 将坐标列表中的字符串转换为整数，并分别赋值给 x, y, z
            x, y, z = (int(num) for num in xyz)
            # 返回转换后的坐标
            return x, y, z
        # 如果出现数值错误，提示用户只能输入整数
        except ValueError:
            print("Please enter whole numbers only")
# 定义一个函数，用于玩游戏，接受搜索区域和深度炸弹数量作为参数
def play_game(search_area, num_charges) -> None:
    # 打印游戏开始提示信息
    print("\nYou are the captain of the destroyer USS Computer.")
    print("An enemy sub has been causing you trouble. Your")
    print(f"mission is to destroy it. You have {num_charges} shots.")
    print("Specify depth charge explosion point with a")
    print("trio of numbers -- the first two are the")
    print("surface coordinates; the third is the depth.")
    print("\nGood luck!\n")

    # 生成潜艇的位置
    a, b, c = (random.randint(0, search_area) for _ in range(3))

    # 循环获取玩家输入，直到胜利或失败
    for i in range(num_charges):
        print(f"\nTrial #{i+1}")
        x, y, z = get_shot_input()

        # 判断玩家输入的炸弹位置是否与潜艇位置相同
        if (x, y, z) == (a, b, c):
            print(f"\nB O O M ! ! You found it in {i+1} tries!\n")
            ask_for_new_game()
        else:
            show_shot_result((x, y, z), (a, b, c))

    # 炸弹用尽，游戏结束
    print("\nYou have been torpedoed! Abandon ship!")
    print(f"The submarine was at {a} {b} {c}")
    ask_for_new_game()

# 定义一个函数，用于获取搜索区域和深度炸弹数量，并调用play_game函数开始游戏
def main() -> None:
    search_area, num_charges = get_num_charges()
    play_game(search_area, num_charges)

# 如果当前脚本为主程序，则调用main函数开始游戏
if __name__ == "__main__":
    main()
```