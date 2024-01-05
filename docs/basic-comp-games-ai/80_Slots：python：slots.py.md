# `80_Slots\python\slots.py`

```
这部分代码是一个注释块，用于说明程序的功能和背景信息。它描述了老虎机的工作原理和历史，以及一些相关的历史背景信息。这些注释并不直接与代码功能相关，而是提供了一些背景知识。
import sys  # 导入sys模块，用于访问与Python解释器交互的变量和函数
from collections import Counter  # 导入Counter类，用于计数可哈希对象
from random import choices  # 导入choices函数，用于从给定的序列中随机选择元素
from typing import List  # 导入List类型，用于声明列表类型

def initial_message() -> None:  # 定义initial_message函数，返回类型为None
    # 打印空格和"Slots"字符串
    print(" " * 30 + "Slots")
    # 打印空格和"Creative Computing Morrison, New Jersey"字符串
    print(" " * 15 + "Creative Computing Morrison, New Jersey")
    # 打印三个换行符
    print("\n" * 3)
    # 打印"You are in the H&M Casino, in front of one of our"字符串
    print("You are in the H&M Casino, in front of one of our")
    # 打印"one-arm Bandits. Bet from $1 to $100."字符串
    print("one-arm Bandits. Bet from $1 to $100.")
    # 打印"To pull the arm, punch the return key after making your bet."字符串
    print("To pull the arm, punch the return key after making your bet.")


def input_betting() -> int:
    # 打印换行符
    print("\n")
    # 初始化变量b为-1
    b = -1
    # 当b小于1或者大于100时循环
    while b < 1 or b > 100:
        try:
            # 尝试从用户输入中获取整数赋值给b
            b = int(input("Your bet:"))
        except ValueError:
            # 如果输入不是整数，则将b设为-1
            b = -1
        if b > 100:
            # 如果b大于100，则打印"House limits are $100"
            print("House limits are $100")
        elif b < 1:
            # 如果b小于1，则打印"Minium bet is $1"
            print("Minium bet is $1")
    beeping()  # 调用名为beeping的函数，产生蜂鸣声
    return int(b)  # 返回变量b的整数值


def beeping() -> None:
    # 产生蜂鸣声的函数
    # 在原始程序中是第1270行的子程序
    for _ in range(5):  # 循环5次
        sys.stdout.write("\a")  # 输出控制台响铃符号
        sys.stdout.flush()  # 刷新控制台


def spin_wheels() -> List[str]:
    possible_fruits = ["Bar", "Bell", "Orange", "Lemon", "Plum", "Cherry"]  # 可能的水果列表
    wheel = choices(possible_fruits, k=3)  # 从可能的水果列表中随机选择3个水果

    print(*wheel)  # 打印出选中的3个水果
    beeping()  # 调用名为beeping的函数，产生蜂鸣声

    return wheel  # 返回选中的3个水果
# 根据轮盘上的水果列表、下注金额和当前利润调整利润
def adjust_profits(wheel: List[str], m: int, profits: int) -> int:
    # 移除重复的水果
    s = set(wheel)

    if len(s) == 1:
        # 三种水果都相同
        fruit = s.pop()

        if fruit == "Bar":
            print("\n***Jackpot***")
            profits = ((100 * m) + m) + profits  # 利润增加
        else:
            print("\n**Top Dollar**")
            profits = ((10 * m) + m) + profits  # 利润增加

        print("You Won!")  # 打印赢得奖励的消息
    elif len(s) == 2:
        # 两种水果相同
        c = Counter(wheel)  # 使用 Counter 对象统计轮子上水果出现的次数
        # we get the fruit that appears two times
        fruit = sorted(c.items(), key=lambda x: x[1], reverse=True)[0][0]  # 找出出现次数最多的水果

        if fruit == "Bar":  # 如果出现次数最多的水果是 "Bar"
            print("\n*Double Bar*")  # 打印消息
            profits = ((5 * m) + m) + profits  # 更新利润
        else:  # 如果出现次数最多的水果不是 "Bar"
            print("\nDouble!!")  # 打印消息
            profits = ((2 * m) + m) + profits  # 更新利润

        print("You Won!")  # 打印消息
    else:  # 如果轮子上有三种不同的水果
        # three different fruits
        print("\nYou Lost.")  # 打印消息
        profits = profits - m  # 更新利润

    return profits  # 返回最终利润
def final_message(profits: int) -> None:
    # 如果利润小于0，打印“付钱！请把你的钱留在终端上”
    if profits < 0:
        print("Pay up!  Please leave your money on the terminal")
    # 如果利润等于0，打印“嘿，你打平了。”
    elif profits == 0:
        print("Hey, You broke even.")
    # 否则，打印“从H&M收银员那里领取你的赢利。”
    else:
        print("Collect your winings from the H&M cashier.")


def main() -> None:
    profits = 0
    keep_betting = True

    initial_message()  # 调用初始消息函数
    while keep_betting:  # 当继续下注为真时执行循环
        m = input_betting()  # 获取下注金额
        w = spin_wheels()  # 旋转轮子
        profits = adjust_profits(w, m, profits)  # 调整利润
        print(f"Your standings are ${profits}")  # 打印当前利润
        answer = input("Again?")  # 询问用户是否要再次执行程序，并将用户输入的内容存储在变量answer中

        try:
            if answer[0].lower() != "y":  # 如果用户输入的第一个字符不是小写的y
                keep_betting = False  # 将keep_betting变量设为False，表示不再进行下注
        except IndexError:  # 如果用户没有输入任何内容
            keep_betting = False  # 将keep_betting变量设为False，表示不再进行下注

    final_message(profits)  # 调用final_message函数，传入profits参数

if __name__ == "__main__":  # 如果当前脚本被直接执行
    main()  # 调用main函数

######################################################################
#
# Porting notes
#
#   The selections of the fruits(Bar, apples, lemon, etc.) are made
#   with equal probability, accordingly to random.choices documentation.
```
```python
# Porting notes
#
#   The selections of the fruits(Bar, apples, lemon, etc.) are made
#   with equal probability, accordingly to random.choices documentation.
```
```python
# Porting notes
#
#   The selections of the fruits(Bar, apples, lemon, etc.) are made
#   with equal probability, accordingly to random.choices documentation.
```
```python
# Porting notes
#
#   The selections of the fruits(Bar, apples, lemon, etc.) are made
#   with equal probability, accordingly to random.choices documentation.
# 这段代码是一个注释，说明可以向函数添加一个权重列表，以调整预期的回报。注释的作用是对函数的功能进行补充说明，提供了一个可能的改进方向。
```