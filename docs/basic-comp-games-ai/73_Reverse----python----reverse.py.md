# `basic-computer-games\73_Reverse\python\reverse.py`

```

#!/usr/bin/env python3
# 指定使用 Python3 解释器

import random
import textwrap
# 导入 random 和 textwrap 模块

NUMCNT = 9  # How many numbers are we playing with?
# 定义全局变量 NUMCNT，表示游戏中使用的数字个数

def main() -> None:
    # 主函数，打印游戏标题和规则，然后进入游戏循环
    print("REVERSE".center(72))
    print("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".center(72))
    print()
    print()
    print("REVERSE -- A GAME OF SKILL")
    print()

    if not input("DO YOU WANT THE RULES? (yes/no) ").lower().startswith("n"):
        print_rules()
    # 如果用户需要规则，则打印规则

    while True:
        game_loop()
        # 进入游戏循环

        if not input("TRY AGAIN? (yes/no) ").lower().startswith("y"):
            return
        # 如果用户不想再玩了，则退出游戏

def game_loop() -> None:
    """Play the main game."""
    # 主游戏循环函数
    numbers = list(range(1, NUMCNT + 1))
    random.shuffle(numbers)
    # 生成一个包含 1 到 NUMCNT 的随机排列列表

    print()
    print("HERE WE GO ... THE LIST IS:")
    print_list(numbers)
    # 打印初始列表

    turns = 0
    while True:
        try:
            howmany = int(input("HOW MANY SHALL I REVERSE? "))
            assert howmany >= 0
        except (ValueError, AssertionError):
            continue
        # 获取用户输入的翻转数量，确保输入合法

        if howmany == 0:
            return
        # 如果用户输入 0，则退出游戏

        if howmany > NUMCNT:
            print("OOPS! WRONG! I CAN REVERSE AT MOST", NUMCNT)
            continue
        # 如果用户输入的翻转数量超过了 NUMCNT，则提示错误

        turns += 1

        newnums = numbers[0:howmany]
        newnums.reverse()
        newnums.extend(numbers[howmany:])
        numbers = newnums
        # 翻转列表中指定数量的数字

        print_list(numbers)
        # 打印翻转后的列表

        if all(numbers[i] == i + 1 for i in range(NUMCNT)):
            print(f"YOU WON IT IN {turns} MOVES!")
            print()
            return
        # 检查是否获胜，如果是则打印获胜信息并退出游戏

def print_list(numbers) -> None:
    print(" ".join(map(str, numbers))
    # 打印列表

def print_rules() -> None:
    # 打印游戏规则
    help = textwrap.dedent(
        """
        THIS IS THE GAME OF "REVERSE".  TO WIN, ALL YOU HAVE
        TO DO IS ARRANGE A LIST OF NUMBERS (1 THROUGH {})
        IN NUMERICAL ORDER FROM LEFT TO RIGHT.  TO MOVE, YOU
        TELL ME HOW MANY NUMBERS (COUNTING FROM THE LEFT) TO
        REVERSE.  FOR EXAMPLE, IF THE CURRENT LIST IS:

        2 3 4 5 1 6 7 8 9

        AND YOU REVERSE 4, THE RESULT WILL BE:

        5 4 3 2 1 6 7 8 9

        NOW IF YOU REVERSE 5, YOU WIN!

        1 2 3 4 5 6 7 8 9

        NO DOUBT YOU WILL LIKE THIS GAME, BUT
        IF YOU WANT TO QUIT, REVERSE 0 (ZERO).
        """.format(
            NUMCNT
        )
    )
    print(help)
    print()
    # 打印游戏规则

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    # 如果程序被中断，则捕获异常并退出```
```