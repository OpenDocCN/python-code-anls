# `basic-computer-games\45_Hello\python\hello.py`

```

"""
HELLO

A very simple "chat" bot.

Warning, the advice given here is bad.

Ported by Dave LeCompte
"""

import time  # 导入时间模块
from typing import Optional, Tuple  # 导入类型提示模块


def get_yes_or_no() -> Tuple[bool, Optional[bool], str]:  # 定义函数，返回布尔值、可选布尔值和字符串的元组
    msg = input()  # 获取用户输入
    if msg.upper() == "YES":  # 如果用户输入为"YES"
        return True, True, msg  # 返回True, True, 用户输入
    elif msg.upper() == "NO":  # 如果用户输入为"NO"
        return True, False, msg  # 返回True, False, 用户输入
    else:  # 其他情况
        return False, None, msg  # 返回False, None, 用户输入


def ask_enjoy_question(user_name: str) -> None:  # 定义函数，接受用户名参数，不返回任何内容
    print(f"HI THERE, {user_name}, ARE YOU ENJOYING YOURSELF HERE?")  # 打印问候语

    while True:  # 无限循环
        valid, value, msg = get_yes_or_no()  # 调用函数获取用户输入

        if valid:  # 如果输入有效
            if value:  # 如果值为True
                print(f"I'M GLAD TO HEAR THAT, {user_name}.")  # 打印肯定回答
                print()
            else:  # 如果值为False
                print(f"OH, I'M SORRY TO HEAR THAT, {user_name}. MAYBE WE CAN")  # 打印否定回答
                print("BRIGHTEN UP YOUR VISIT A BIT.")
            break  # 结束循环
        else:  # 如果输入无效
            print(f"{user_name}, I DON'T UNDERSTAND YOUR ANSWER OF '{msg}'.")  # 打印提示
            print("PLEASE ANSWER 'YES' OR 'NO'.  DO YOU LIKE IT HERE?")  # 打印提示


# 其他函数的注释省略，均为类似的功能实现
# ...


def main() -> None:  # 定义主函数，不返回任何内容
    print(" " * 33 + "HELLO")  # 打印问候语
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")  # 打印信息
    print("HELLO.  MY NAME IS CREATIVE COMPUTER.\n\n")  # 打印信息
    print("WHAT'S YOUR NAME?")  # 打印提示
    user_name = input()  # 获取用户输入
    print()

    ask_enjoy_question(user_name)  # 调用函数

    ask_question_loop(user_name)  # 调用函数

    ask_for_fee(user_name)  # 调用函数

    if False:  # 如果条件为False
        happy_goodbye(user_name)  # 调用函数
    else:  # 如果条件为True
        unhappy_goodbye(user_name)  # 调用函数


if __name__ == "__main__":  # 如果模块被直接运行
    main()  # 调用主函数

```