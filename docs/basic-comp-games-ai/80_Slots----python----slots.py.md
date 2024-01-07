# `basic-computer-games\80_Slots\python\slots.py`

```

# 导入所需的模块
import sys
from collections import Counter
from random import choices
from typing import List

# 打印初始消息
def initial_message() -> None:
    print(" " * 30 + "Slots")
    print(" " * 15 + "Creative Computing Morrison, New Jersey")
    print("\n" * 3)
    print("You are in the H&M Casino, in front of one of our")
    print("one-arm Bandits. Bet from $1 to $100.")
    print("To pull the arm, punch the return key after making your bet.")

# 获取用户下注金额
def input_betting() -> int:
    print("\n")
    b = -1
    while b < 1 or b > 100:
        try:
            b = int(input("Your bet:"))
        except ValueError:
            b = -1
        if b > 100:
            print("House limits are $100")
        elif b < 1:
            print("Minium bet is $1")
    beeping()
    return int(b)

# 产生蜂鸣声
def beeping() -> None:
    # 产生蜂鸣声的函数
    # 在原始程序中是第1270行的子程序
    for _ in range(5):
        sys.stdout.write("\a")
        sys.stdout.flush()

# 旋转轮子并返回结果
def spin_wheels() -> List[str]:
    possible_fruits = ["Bar", "Bell", "Orange", "Lemon", "Plum", "Cherry"]
    wheel = choices(possible_fruits, k=3)

    print(*wheel)
    beeping()

    return wheel

# 调整利润
def adjust_profits(wheel: List[str], m: int, profits: int) -> int:
    # 移除重复的水果
    s = set(wheel)

    if len(s) == 1:
        # 三个水果相同
        fruit = s.pop()

        if fruit == "Bar":
            print("\n***Jackpot***")
            profits = ((100 * m) + m) + profits
        else:
            print("\n**Top Dollar**")
            profits = ((10 * m) + m) + profits

        print("You Won!")
    elif len(s) == 2:
        # 两个水果相同
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

    return profits

# 打印最终消息
def final_message(profits: int) -> None:
    if profits < 0:
        print("Pay up!  Please leave your money on the terminal")
    elif profits == 0:
        print("Hey, You broke even.")
    else:
        print("Collect your winings from the H&M cashier.")

# 主函数
def main() -> None:
    profits = 0
    keep_betting = True

    initial_message()
    while keep_betting:
        m = input_betting()
        w = spin_wheels()
        profits = adjust_profits(w, m, profits)

        print(f"Your standings are ${profits}")
        answer = input("Again?")

        try:
            if answer[0].lower() != "y":
                keep_betting = False
        except IndexError:
            keep_betting = False

    final_message(profits)

# 如果是直接运行该脚本，则执行主函数
if __name__ == "__main__":
    main()

```