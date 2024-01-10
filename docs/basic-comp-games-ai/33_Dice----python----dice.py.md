# `basic-computer-games\33_Dice\python\dice.py`

```
"""
Dice

From: BASIC Computer Games (1978)
      Edited by David H. Ahl

"Not exactly a game, this program simulates rolling
 a pair of dice a large number of times and prints out
 the frequency distribution.  You simply input the
 number of rolls.  It is interesting to see how many
 rolls are necessary to approach the theoretical
 distribution:

 2  1/36  2.7777...%
 3  2/36  5.5555...%
 4  3/36  8.3333...%
   etc.

"Daniel Freidus wrote this program while in the
 seventh grade at Harrison Jr-Sr High School,
 Harrison, New York."

Python port by Jeff Jetton, 2019
"""

import random


def main() -> None:
    # We'll track counts of roll outcomes in a 13-element list.
    # The first two indices (0 & 1) are ignored, leaving just
    # the indices that match the roll values (2 through 12).
    freq = [0] * 13

    # Display intro text
    print("\n                   Dice")
    print("Creative Computing  Morristown, New Jersey")
    print("\n\n")
    # "Danny Freidus"
    print("This program simulates the rolling of a")
    print("pair of dice.")
    print("You enter the number of times you want the computer to")
    print("'roll' the dice.   Watch out, very large numbers take")
    print("a long time.  In particular, numbers over 5000.")

    still_playing = True
    # 当仍在玩游戏时执行以下操作
    while still_playing:
        # 打印空行
        print()
        # 获取用户输入的掷骰子次数
        n = int(input("How many rolls? "))

        # 掷骰子 n 次
        for _ in range(n):
            # 随机生成两个骰子的点数
            die1 = random.randint(1, 6)
            die2 = random.randint(1, 6)
            # 计算两个骰子的点数总和
            roll_total = die1 + die2
            # 更新点数总和的频率
            freq[roll_total] += 1

        # 显示最终结果
        print("\nTotal Spots   Number of Times")
        for i in range(2, 13):
            # 打印点数总和及其出现次数
            print(" %-14d%d" % (i, freq[i]))

        # 继续游戏？
        print()
        response = input("Try again? ")
        # 如果用户输入的是以 Y 开头的字符串，则执行以下操作
        if len(response) > 0 and response.upper()[0] == "Y":
            # 清空频率列表
            freq = [0] * 13
        else:
            # 退出游戏循环
            still_playing = False
# 如果当前脚本被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()

########################################################
#
# Porting Notes
#
#   一个相当直接的移植。唯一的变化在于处理用户的“再试一次”响应。
#   原始程序只有在用户输入“YES”时才继续，而这个版本将在给出以“Y”或“y”开头的任何单词时继续。
#
#   指令文本--像所有这些移植一样，都是从原始清单中逐字摘录--在对设置摇动次数过高发出严重警告时，具有迷人的古雅风味。
#   在撰写本文时，一个相当慢的计算机上，5000次摇动通常在不到1/10秒的时间内完成！
#
#
# Ideas for Modifications
#
#   让结果包括第三列，显示每个计数所代表的百分比。或者（更好的是）使用星号的行来打印低保真条形图，表示相对值，每个星号代表1%，
#   例如。
#
#   添加一列显示理论上预期的百分比，以供比较。
#
#   跟踪一系列摇动所花费的时间，并将该信息添加到最终报告中。
#
#   如果每次摇动三个（或四个，或五个...）骰子会怎样？
#
########################################################
```