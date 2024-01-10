# `basic-computer-games\92_Trap\python\trap.py`

```
#!/usr/bin/env python3
# 指定脚本解释器为 Python 3

# TRAP
#
# STEVE ULLMAN, 8-1-72
# Converted from BASIC to Python by Trevor Hobson
# 游戏的作者和转换者信息

import random
# 导入 random 模块

number_max = 100
# 定义最大数字为 100
guess_max = 6
# 定义最大猜测次数为 6


def play_game() -> None:
    """Play one round of the game"""
    # 定义一个函数，用于进行一轮游戏

    number_computer = random.randint(1, number_max)
    # 生成一个随机数作为电脑选择的数字
    turn = 0
    # 初始化猜测次数为 0
    while True:
        # 进入循环
        turn += 1
        # 猜测次数加一
        user_guess = [-1, -1]
        # 初始化用户猜测的列表
        while user_guess == [-1, -1]:
            # 进入循环，直到用户输入有效的猜测
            try:
                user_input = [
                    int(item)
                    for item in input("\nGuess # " + str(turn) + " ? ").split(",")
                ]
                # 获取用户输入的猜测
                if len(user_input) == 2:
                    # 如果用户输入了两个数字
                    if sum(1 < x < number_max for x in user_input) == 2:
                        # 如果两个数字都在有效范围内
                        user_guess = user_input
                        # 将用户猜测设置为用户输入的值
                    else:
                        raise ValueError
                        # 抛出值错误异常
                else:
                    raise ValueError
                    # 抛出值错误异常
            except (ValueError, IndexError):
                # 捕获值错误和索引错误异常
                print("Please enter a valid guess.")
                # 提示用户输入有效的猜测
        if user_guess[0] > user_guess[1]:
            user_guess[0], user_guess[1] = user_guess[1], user_guess[0]
            # 如果用户输入的第一个数字大于第二个数字，交换两个数字的位置
        if user_guess[0] == user_guess[1] == number_computer:
            print("You got it!!!")
            # 如果用户猜中了电脑选择的数字，输出提示信息并结束游戏
            break
        elif user_guess[0] <= number_computer <= user_guess[1]:
            print("You have trapped my number.")
            # 如果用户的猜测范围包含了电脑选择的数字，输出提示信息
        elif number_computer < user_guess[0]:
            print("My number is smaller than your trap numbers.")
            # 如果电脑选择的数字小于用户的猜测范围，输出提示信息
        else:
            print("My number is larger than your trap numbers.")
            # 如果电脑选择的数字大于用户的猜测范围，输出提示信息
        if turn == guess_max:
            print("That's", turn, "guesses. The number was", number_computer)
            # 如果猜测次数达到最大次数，输出提示信息并结束游戏
            break


def main() -> None:
    print(" " * 34 + "TRAP")
    # 输出游戏标题
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    # 输出创意计算的信息
    # 如果用户输入以字母y开头的字符串，则执行以下操作
    if input("Instructions ").lower().startswith("y"):
        # 打印游戏提示信息
        print("\nI am thinking of a number between 1 and", number_max)
        print("try to guess my number. On each guess,")
        print("you are to enter 2 numbers, trying to trap")
        print("my number between the two numbers. I will")
        print("tell you if you have trapped my number, if my")
        print("number is larger than your two numbers, or if")
        print("my number is smaller than your two numbers.")
        print("If you want to guess one single number, type")
        print("your guess for both your trap numbers.")
        print("You get", guess_max, "guesses to get my number.")
    
    # 设置一个变量用于控制是否继续游戏
    keep_playing = True
    # 当keep_playing为True时，执行以下操作
    while keep_playing:
        # 调用play_game()函数进行游戏
        play_game()
        # 根据用户输入判断是否继续游戏
        keep_playing = input("\nTry again. ").lower().startswith("y")
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```