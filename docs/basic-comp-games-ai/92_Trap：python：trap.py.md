# `d:/src/tocomm/basic-computer-games\92_Trap\python\trap.py`

```
#!/usr/bin/env python3
# TRAP
#
# STEVE ULLMAN, 8-1-72
# Converted from BASIC to Python by Trevor Hobson

import random

number_max = 100  # 设置最大猜测数字为100
guess_max = 6  # 设置最大猜测次数为6


def play_game() -> None:
    """Play one round of the game"""

    number_computer = random.randint(1, number_max)  # 生成一个1到100之间的随机整数作为计算机选择的数字
    turn = 0  # 初始化猜测次数为0
    while True:  # 进入循环
        turn += 1  # 每次循环猜测次数加1
        user_guess = [-1, -1]  # 初始化用户猜测的数字为[-1, -1]
        while user_guess == [-1, -1]:  # 当用户猜测为[-1, -1]时循环执行以下代码
            try:  # 尝试执行以下代码
                user_input = [  # 将用户输入的字符串按逗号分割后转换为整数列表
                    int(item)
                    for item in input("\nGuess # " + str(turn) + " ? ").split(",")
                ]
                if len(user_input) == 2:  # 如果用户输入的列表长度为2
                    if sum(1 < x < number_max for x in user_input) == 2:  # 如果用户输入的两个数都在1和number_max之间
                        user_guess = user_input  # 将用户输入的列表赋值给user_guess
                    else:  # 如果用户输入的两个数不在1和number_max之间
                        raise ValueError  # 抛出值错误异常
                else:  # 如果用户输入的列表长度不为2
                    raise ValueError  # 抛出值错误异常
            except (ValueError, IndexError):  # 捕获值错误或索引错误异常
                print("Please enter a valid guess.")  # 打印提示信息
        if user_guess[0] > user_guess[1]:  # 如果用户猜测的第一个数大于第二个数
            user_guess[0], user_guess[1] = user_guess[1], user_guess[0]  # 交换两个数的位置
        if user_guess[0] == user_guess[1] == number_computer:  # 如果用户猜测的两个数都等于number_computer
            print("You got it!!!")  # 打印提示信息
            break  # 跳出循环
        elif user_guess[0] <= number_computer <= user_guess[1]:  # 如果用户猜测的范围包含了计算机生成的数字
            print("You have trapped my number.")  # 打印出用户已经成功捕捉到计算机生成的数字
        elif number_computer < user_guess[0]:  # 如果计算机生成的数字小于用户猜测的范围的最小值
            print("My number is smaller than your trap numbers.")  # 打印出计算机生成的数字比用户猜测的范围的最小值还要小
        else:  # 如果计算机生成的数字大于用户猜测的范围的最大值
            print("My number is larger than your trap numbers.")  # 打印出计算机生成的数字比用户猜测的范围的最大值还要大
        if turn == guess_max:  # 如果轮次等于最大猜测次数
            print("That's", turn, "guesses. The number was", number_computer)  # 打印出已经猜测了多少次以及计算机生成的数字是多少
            break  # 结束循环

def main() -> None:  # 主函数声明，不返回任何值
    print(" " * 34 + "TRAP")  # 打印出"TRAP"
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")  # 打印出"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并换行
    if input("Instructions ").lower().startswith("y"):  # 如果用户输入的指令以"y"开头
        print("\nI am thinking of a number between 1 and", number_max)  # 打印出计算机正在想一个1到number_max之间的数字
        print("try to guess my number. On each guess,")  # 打印出"try to guess my number. On each guess,"
        print("you are to enter 2 numbers, trying to trap")  # 打印出"you are to enter 2 numbers, trying to trap"
        print("my number between the two numbers. I will")  # 打印出"my number between the two numbers. I will"
        print("tell you if you have trapped my number, if my")  # 打印出"tell you if you have trapped my number, if my"
        print("number is larger than your two numbers, or if")  # 打印提示信息
        print("my number is smaller than your two numbers.")  # 打印提示信息
        print("If you want to guess one single number, type")  # 打印提示信息
        print("your guess for both your trap numbers.")  # 打印提示信息
        print("You get", guess_max, "guesses to get my number.")  # 打印提示信息，guess_max为变量

    keep_playing = True  # 初始化变量keep_playing为True
    while keep_playing:  # 当keep_playing为True时执行循环
        play_game()  # 调用play_game函数
        keep_playing = input("\nTry again. ").lower().startswith("y")  # 获取用户输入，如果以"y"开头则将keep_playing设置为True，否则设置为False


if __name__ == "__main__":  # 如果当前文件被直接运行
    main()  # 调用main函数
```