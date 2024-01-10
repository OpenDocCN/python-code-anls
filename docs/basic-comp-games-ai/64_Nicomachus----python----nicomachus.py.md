# `basic-computer-games\64_Nicomachus\python\nicomachus.py`

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
    # 循环直到得到有效的 YES 或 NO 回答
    while True:
        # 获取用户输入并转换为大写
        response = input().upper()
        # 如果回答是 "YES"，返回 True
        if response == "YES":
            return True
        # 如果回答是 "NO"，返回 False
        elif response == "NO":
            return False
        # 如果回答既不是 "YES" 也不是 "NO"，提示用户重新输入
        print(f"EH?  I DON'T UNDERSTAND '{response}'  TRY 'YES' OR 'NO'.")


def play_game() -> None:
    # 提示用户思考一个 1 到 100 之间的数字
    print("PLEASE THINK OF A NUMBER BETWEEN 1 AND 100.")
    print("YOUR NUMBER DIVIDED BY 3 HAS A REMAINDER OF")
    # 获取用户输入的数字
    a = int(input())
    print("YOUR NUMBER DIVIDED BY 5 HAS A REMAINDER OF")
    # 获取用户输入的数字
    b = int(input())
    print("YOUR NUMBER DIVIDED BY 7 HAS A REMAINDER OF")
    # 获取用户输入的数字
    c = int(input())
    print()
    print("LET ME THINK A MOMENT...")
    print()

    # 程序休眠 2.5 秒
    time.sleep(2.5)

    # 计算用户所想的数字
    d = (70 * a + 21 * b + 15 * c) % 105

    # 输出程序猜测的数字
    print(f"YOUR NUMBER WAS {d}, RIGHT?")

    # 获取用户对猜测结果的回答
    response = get_yes_or_no()

    # 根据用户回答输出不同的结果
    if response:
        print("HOW ABOUT THAT!!")
    else:
        print("I FEEL YOUR ARITHMETIC IS IN ERROR.")
    print()
    print("LET'S TRY ANOTHER")


def main() -> None:
    # 输出游戏标题
    print(" " * 33 + "NICOMA")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

    # 输出游戏介绍
    print("BOOMERANG PUZZLE FROM ARITHMETICA OF NICOMACHUS -- A.D. 90!")
    print()
    # 循环进行游戏
    while True:
        play_game()


if __name__ == "__main__":
    main()
```