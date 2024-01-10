# `basic-computer-games\80_Slots\python\slots.py`

```
# 导入 sys 模块
import sys
# 导入 Counter 类
from collections import Counter
# 从 random 模块中导入 choices 函数
from random import choices
# 从 typing 模块中导入 List 类型
from typing import List

# 定义 initial_message 函数，无返回值
def initial_message() -> None:
    # 打印游戏标题
    print(" " * 30 + "Slots")
    # 打印游戏信息
    print(" " * 15 + "Creative Computing Morrison, New Jersey")
    print("\n" * 3)
    # 打印赌场信息和操作提示
    print("You are in the H&M Casino, in front of one of our")
    print("one-arm Bandits. Bet from $1 to $100.")
    print("To pull the arm, punch the return key after making your bet.")

# 定义 input_betting 函数，返回值为整数类型
def input_betting() -> int:
    # 初始化赌注变量为 -1
    b = -1
    # 当下注金额小于1或大于100时，进入循环
    while b < 1 or b > 100:
        # 尝试获取用户输入的下注金额，如果输入不是整数则抛出异常
        try:
            b = int(input("Your bet:"))
        except ValueError:
            # 如果输入不是整数，则将下注金额设为-1
            b = -1
        # 如果下注金额大于100，则打印提示信息
        if b > 100:
            print("House limits are $100")
        # 如果下注金额小于1，则打印提示信息
        elif b < 1:
            print("Minium bet is $1")
    # 调用beeping函数
    beeping()
    # 返回整数类型的下注金额
    return int(b)
# 定义一个产生蜂鸣声的函数，没有返回值
def beeping() -> None:
    # 产生蜂鸣声的循环，共产生5次蜂鸣
    for _ in range(5):
        # 输出蜂鸣符号
        sys.stdout.write("\a")
        # 刷新标准输出缓冲区
        sys.stdout.flush()


# 旋转轮子的函数，返回一个字符串列表
def spin_wheels() -> List[str]:
    # 可能的水果列表
    possible_fruits = ["Bar", "Bell", "Orange", "Lemon", "Plum", "Cherry"]
    # 从可能的水果列表中随机选择3个水果
    wheel = choices(possible_fruits, k=3)

    # 打印轮子上的水果
    print(*wheel)
    # 调用产生蜂鸣声的函数
    beeping()

    # 返回轮子上的水果列表
    return wheel


# 调整利润的函数，接受轮子上的水果列表、赌注和利润作为参数，返回一个整数
def adjust_profits(wheel: List[str], m: int, profits: int) -> int:
    # 去除重复的水果
    s = set(wheel)

    if len(s) == 1:
        # 三个水果都相同
        fruit = s.pop()

        if fruit == "Bar":
            print("\n***Jackpot***")
            profits = ((100 * m) + m) + profits
        else:
            print("\n**Top Dollar**")
            profits = ((10 * m) + m) + profits

        print("You Won!")
    elif len(s) == 2:
        # 有两个水果相同
        c = Counter(wheel)
        # 获取出现两次的水果
        fruit = sorted(c.items(), key=lambda x: x[1], reverse=True)[0][0]

        if fruit == "Bar":
            print("\n*Double Bar*")
            profits = ((5 * m) + m) + profits
        else:
            print("\nDouble!!")
            profits = ((2 * m) + m) + profits

        print("You Won!")
    else:
        # 三个不同的水果
        print("\nYou Lost.")
        profits = profits - m

    # 返回调整后的利润
    return profits


# 最终消息的函数，接受利润作为参数，没有返回值
def final_message(profits: int) -> None:
    if profits < 0:
        print("Pay up!  Please leave your money on the terminal")
    elif profits == 0:
        print("Hey, You broke even.")
    else:
        print("Collect your winings from the H&M cashier.")


# 主函数
def main() -> None:
    # 初始化利润为0
    profits = 0
    # 继续下注的标志
    keep_betting = True

    # 调用初始消息的函数
    initial_message()
    # 当继续下注时执行循环
    while keep_betting:
        # 获取用户下注金额
        m = input_betting()
        # 旋转轮盘，获取结果
        w = spin_wheels()
        # 调整利润
        profits = adjust_profits(w, m, profits)
    
        # 打印当前利润
        print(f"Your standings are ${profits}")
        # 询问用户是否继续下注
        answer = input("Again?")
    
        # 尝试检查用户输入的第一个字符是否为小写的 "y"，如果不是则停止下注
        try:
            if answer[0].lower() != "y":
                keep_betting = False
        # 处理用户没有输入的情况，停止下注
        except IndexError:
            keep_betting = False
    
    # 执行游戏结束的消息
    final_message(profits)
# 如果当前模块被直接执行，则调用 main 函数
if __name__ == "__main__":
    main()

######################################################################
#
# 移植说明
#
#   水果的选择（Bar, apples, lemon, 等）是根据 random.choices 文档中的等概率进行的。
#   可以向函数添加一个权重列表，从而调整预期的回报
#
######################################################################
```