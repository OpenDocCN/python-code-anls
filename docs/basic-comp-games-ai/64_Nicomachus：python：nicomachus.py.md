# `d:/src/tocomm/basic-computer-games\64_Nicomachus\python\nicomachus.py`

```
"""
NICOMACHUS

Math exercise/demonstration

Ported by Dave LeCompte
"""

# PORTING NOTE
#
# The title, as printed ingame, is "NICOMA", hinting at a time when
# filesystems weren't even 8.3, but could only support 6 character
# filenames.

import time


def get_yes_or_no() -> bool:
    # 无限循环，直到用户输入了合法的是/否回答
    while True:
        # 获取用户输入并转换为大写
        response = input().upper()
        if response == "YES":  # 如果用户输入的回答是"YES"
            return True  # 返回True
        elif response == "NO":  # 如果用户输入的回答是"NO"
            return False  # 返回False
        print(f"EH?  I DON'T UNDERSTAND '{response}'  TRY 'YES' OR 'NO'.")  # 如果用户输入的回答既不是"YES"也不是"NO"，则打印提示信息


def play_game() -> None:  # 定义一个名为play_game的函数，不返回任何结果
    print("PLEASE THINK OF A NUMBER BETWEEN 1 AND 100.")  # 打印提示信息
    print("YOUR NUMBER DIVIDED BY 3 HAS A REMAINDER OF")  # 打印提示信息
    a = int(input())  # 获取用户输入的整数并赋值给变量a
    print("YOUR NUMBER DIVIDED BY 5 HAS A REMAINDER OF")  # 打印提示信息
    b = int(input())  # 获取用户输入的整数并赋值给变量b
    print("YOUR NUMBER DIVIDED BY 7 HAS A REMAINDER OF")  # 打印提示信息
    c = int(input())  # 获取用户输入的整数并赋值给变量c
    print()  # 打印空行
    print("LET ME THINK A MOMENT...")  # 打印提示信息
    print()  # 打印空行

    time.sleep(2.5)  # 程序暂停2.5秒
    d = (70 * a + 21 * b + 15 * c) % 105  # 计算数学表达式的结果并赋值给变量d

    print(f"YOUR NUMBER WAS {d}, RIGHT?")  # 打印出变量d的值

    response = get_yes_or_no()  # 调用函数获取用户输入的是或否

    if response:  # 如果用户输入是
        print("HOW ABOUT THAT!!")  # 打印消息
    else:  # 如果用户输入否
        print("I FEEL YOUR ARITHMETIC IS IN ERROR.")  # 打印消息
    print()  # 打印空行
    print("LET'S TRY ANOTHER")  # 打印消息


def main() -> None:  # 定义一个主函数，不返回任何值
    print(" " * 33 + "NICOMA")  # 打印消息
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")  # 打印消息

    print("BOOMERANG PUZZLE FROM ARITHMETICA OF NICOMACHUS -- A.D. 90!")  # 打印消息
    print()  # 打印空行
    while True:  # 无限循环
        play_game()  # 调用 play_game() 函数进行游戏


if __name__ == "__main__":  # 如果当前脚本被直接执行，而不是被导入
    main()  # 调用 main() 函数作为程序的入口点
```