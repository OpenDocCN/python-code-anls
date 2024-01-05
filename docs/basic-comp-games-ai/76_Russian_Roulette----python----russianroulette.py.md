# `76_Russian_Roulette\python\russianroulette.py`

```
"""
Russian Roulette

From Basic Computer Games (1978)

   In this game, you are given by the computer a
  revolver loaded with one bullet and five empty
  chambers. You spin the chamber and pull the trigger
  by inputting a "1", or, if you want to quit, input
  a "2". You win if you play ten times and are still
  alive.
   Tom Adametx wrote this program while a student at
  Curtis Jr. High School in Sudbury, Massachusetts.
"""

# 导入 random 模块中的 random 函数
from random import random

# 定义游戏的轮数
NUMBER_OF_ROUNDS = 9
# 定义 initial_message 函数，打印游戏的初始信息
def initial_message() -> None:
    print(" " * 28 + "Russian Roulette")
    print(" " * 15 + "Creative Computing  Morristown, New Jersey\n\n\n")
    print("This is a game of >>>>>>>>>>Russian Roulette.\n")
    print("Here is a Revolver.")

# 定义 parse_input 函数，用于解析用户输入的整数
def parse_input() -> int:
    while True:
        try:
            i = int(input("? "))  # 获取用户输入的整数
            return i  # 返回用户输入的整数
        except ValueError:
            print("Number expected...")  # 如果用户输入不是整数，则提示用户重新输入

# 定义 main 函数，作为程序的主要逻辑
def main() -> None:
    initial_message()  # 调用 initial_message 函数，打印游戏的初始信息
    while True:  # 进入游戏循环
        # 初始化变量，用于记录是否死亡和轮数
        dead = False
        n = 0
        # 打印游戏提示信息
        print("Type '1' to Spin chamber and pull trigger")
        print("Type '2' to Give up")
        print("Go")
        # 进入游戏循环，直到死亡或放弃
        while not dead:
            # 解析用户输入
            i = parse_input()

            # 如果用户选择放弃，跳出循环
            if i == 2:
                break

            # 如果随机数大于0.8333333333333334，表示死亡
            if random() > 0.8333333333333334:
                dead = True
            else:
                # 否则打印触发器声音，并增加轮数
                print("- CLICK -\n")
                n += 1

            # 如果轮数超过规定的数量，跳出循环
            if n > NUMBER_OF_ROUNDS:
                break
        # 如果死亡，则执行以下代码
        if dead:
# 打印游戏结束的消息
print("BANG!!!!!   You're Dead!")
print("Condolences will be sent to your relatives.\n\n\n")
print("...Next victim...")
# 如果玩家的输入不是1或2，则打印相应消息
else:
    if n > NUMBER_OF_ROUNDS:
        print("You win!!!!!")
        print("Let someone else blow his brain out.\n")
    else:
        print("     Chicken!!!!!\n\n\n")
        print("...Next victim....")


if __name__ == "__main__":
    main()

########################################################
# Porting Notes
#
#    尽管描述说接受"1"或"2"，但原始游戏接受任何数字作为输入，并
# 如果不是"2"，程序会将其视为用户传递了"1"。这个特性在这个版本中被保留了。
# 此外，在原始游戏中，你必须“扣动扳机”11次而不是10次才能赢得游戏，因为在开始时N=0，赢得游戏的条件是“IF N > 10 THEN 80”。这在这个版本中已经修复，只要求用户扣动扳机十次，尽管可以通过更改常量NUMBER_OF_ROUNDS来设置回合数。
```