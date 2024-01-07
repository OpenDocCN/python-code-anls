# `basic-computer-games\76_Russian_Roulette\python\russianroulette.py`

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


from random import random

NUMBER_OF_ROUNDS = 9  # 设置游戏的轮数


def initial_message() -> None:
    print(" " * 28 + "Russian Roulette")
    print(" " * 15 + "Creative Computing  Morristown, New Jersey\n\n\n")
    print("This is a game of >>>>>>>>>>Russian Roulette.\n")
    print("Here is a Revolver.")


def parse_input() -> int:
    while True:
        try:
            i = int(input("? "))  # 获取用户输入的数字
            return i
        except ValueError:
            print("Number expected...")  # 如果输入不是数字，则提示用户重新输入


def main() -> None:
    initial_message()  # 打印游戏初始信息
    while True:
        dead = False
        n = 0
        print("Type '1' to Spin chamber and pull trigger")
        print("Type '2' to Give up")
        print("Go")
        while not dead:
            i = parse_input()  # 获取用户输入

            if i == 2:  # 如果用户输入2，退出游戏
                break

            if random() > 0.8333333333333334:  # 生成随机数，如果大于0.8333333333333334，表示开枪
                dead = True
            else:
                print("- CLICK -\n")  # 否则打印“CLICK”

                n += 1  # 计数器加1

            if n > NUMBER_OF_ROUNDS:  # 如果计数器大于设定的轮数，退出游戏
                break
        if dead:
            print("BANG!!!!!   You're Dead!")  # 打印玩家死亡信息
            print("Condolences will be sent to your relatives.\n\n\n")
            print("...Next victim...")
        else:
            if n > NUMBER_OF_ROUNDS:  # 如果计数器大于设定的轮数，玩家获胜
                print("You win!!!!!")
                print("Let someone else blow his brain out.\n")
            else:
                print("     Chicken!!!!!\n\n\n")  # 否则打印“Chicken”
                print("...Next victim....")


if __name__ == "__main__":
    main()

########################################################
# Porting Notes
#
#    Altough the description says that accepts "1" or "2",
#   the original game accepts any number as input, and
#   if it's different of "2" the program considers
#   as if the user had passed "1". That feature was
#   kept in this port.
#    Also, in the original game you must "pull the trigger"
#   11 times instead of 10 in orden to win,
#   given that N=0 at the beginning and the condition to
#   win is "IF N > 10 THEN  80". That was fixed in this
#   port, asking the user to pull the trigger only ten
#   times, tough the number of round can be set changing
#   the constant NUMBER_OF_ROUNDS.
#
########################################################

注释：以上是一个俄罗斯轮盘赌的游戏程序，根据用户的输入来模拟开枪的过程。程序会根据用户的输入来判断是否开枪，直到达到设定的轮数或者用户选择退出游戏。程序会根据用户的表现来判断输赢，并且有一些额外的说明和注释。
```