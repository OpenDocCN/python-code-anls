# `d:/src/tocomm/basic-computer-games\33_Dice\python\dice.py`

```
# 定义一个名为 Dice 的字符串
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
"""
# We'll track counts of roll outcomes in a 13-element list.
# 创建一个长度为13的列表，用于记录掷骰子的结果频率。前两个索引（0和1）被忽略，只留下与掷骰子值匹配的索引（2到12）。

# The first two indices (0 & 1) are ignored, leaving just
# the indices that match the roll values (2 through 12).
# 前两个索引（0和1）被忽略，只留下与掷骰子值匹配的索引（2到12）。

# Display intro text
# 显示介绍文本
print("\n                   Dice")
print("Creative Computing  Morristown, New Jersey")
print("\n\n")
# "Danny Freidus"
print("This program simulates the rolling of a")
    print("pair of dice.")  # 打印字符串 "pair of dice."
    print("You enter the number of times you want the computer to")  # 打印字符串 "You enter the number of times you want the computer to"
    print("'roll' the dice.   Watch out, very large numbers take")  # 打印字符串 "'roll' the dice.   Watch out, very large numbers take"
    print("a long time.  In particular, numbers over 5000.")  # 打印字符串 "a long time.  In particular, numbers over 5000."

    still_playing = True  # 初始化变量 still_playing 为 True
    while still_playing:  # 进入循环，条件为 still_playing 为 True
        print()  # 打印空行
        n = int(input("How many rolls? "))  # 从用户输入中获取整数值赋给变量 n

        # Roll the dice n times
        for _ in range(n):  # 循环 n 次
            die1 = random.randint(1, 6)  # 生成一个 1 到 6 之间的随机整数并赋给变量 die1
            die2 = random.randint(1, 6)  # 生成一个 1 到 6 之间的随机整数并赋给变量 die2
            roll_total = die1 + die2  # 计算两个骰子的点数总和并赋给变量 roll_total
            freq[roll_total] += 1  # 将点数总和对应的频率加一

        # Display final results
        print("\nTotal Spots   Number of Times")  # 打印字符串 "\nTotal Spots   Number of Times"
        for i in range(2, 13):  # 循环遍历点数总和的可能取值
        print(" %-14d%d" % (i, freq[i]))  # 打印骰子点数和出现次数

        # Keep playing?
        print()  # 打印空行
        response = input("Try again? ")  # 获取用户输入，询问是否继续游戏
        if len(response) > 0 and response.upper()[0] == "Y":  # 如果用户输入以Y开头
            # Clear out the frequency list
            freq = [0] * 13  # 清空骰子点数出现次数列表
        else:
            # Exit the game loop
            still_playing = False  # 退出游戏循环


if __name__ == "__main__":
    main()  # 调用主函数

########################################################
#
# Porting Notes
#
#   一个相当直接的移植。唯一的变化在于处理用户的“再试一次”响应。
#   原始程序只有在用户输入“YES”时才继续，而这个版本将在给出以“Y”或“y”开头的任何单词时继续。
#
#   指令文本--像所有这些移植一样，都是直接从原始清单中摘录的--在对设置滚动次数过高发出严重警告时，具有迷人的古雅风味。在撰写本文时，一个相当慢的计算机上，5000次滚动通常在不到1/10秒的时间内完成！
#
#
# 修改的想法
#
#   使结果包括第三列，显示每个计数所代表的百分比。或者
#   （更好的是）使用低保真条形图打印。
# 用星号行表示相对值，每个星号代表1%
# 例如。
#
# 添加一列显示理论上的预期百分比，以便进行比较。
#
# 跟踪掷骰子的系列所花费的时间，并将该信息添加到最终报告中。
#
# 如果每次掷三个（或四个，或五个...）骰子会怎样？
#
########################################################
```