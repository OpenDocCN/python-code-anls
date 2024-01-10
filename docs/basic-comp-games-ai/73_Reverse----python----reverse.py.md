# `basic-computer-games\73_Reverse\python\reverse.py`

```
#!/usr/bin/env python3
import random
import textwrap

NUMCNT = 9  # How many numbers are we playing with?


def main() -> None:
    # 打印游戏标题
    print("REVERSE".center(72))
    print("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".center(72))
    print()
    print()
    print("REVERSE -- A GAME OF SKILL")
    print()

    # 如果用户想要查看规则，则打印规则
    if not input("DO YOU WANT THE RULES? (yes/no) ").lower().startswith("n"):
        print_rules()

    # 游戏循环
    while True:
        game_loop()

        # 如果用户不想再玩了，则退出游戏
        if not input("TRY AGAIN? (yes/no) ").lower().startswith("y"):
            return


def game_loop() -> None:
    """Play the main game."""
    # 生成1到NUMCNT的随机数列表
    numbers = list(range(1, NUMCNT + 1))
    random.shuffle(numbers)

    # 打印原始列表并开始游戏
    print()
    print("HERE WE GO ... THE LIST IS:")
    print_list(numbers)

    turns = 0
    while True:
        try:
            # 获取用户输入的要反转的数字个数
            howmany = int(input("HOW MANY SHALL I REVERSE? "))
            assert howmany >= 0
        except (ValueError, AssertionError):
            continue

        if howmany == 0:
            return

        if howmany > NUMCNT:
            print("OOPS! WRONG! I CAN REVERSE AT MOST", NUMCNT)
            continue

        turns += 1

        # 反转用户指定数量的数字
        newnums = numbers[0:howmany]
        newnums.reverse()
        newnums.extend(numbers[howmany:])
        numbers = newnums

        print_list(numbers)

        # 检查是否获胜
        if all(numbers[i] == i + 1 for i in range(NUMCNT)):
            print(f"YOU WON IT IN {turns} MOVES!")
            print()
            return


def print_list(numbers) -> None:
    # 打印数字列表
    print(" ".join(map(str, numbers)))


def print_rules() -> None:
    # 打印游戏规则
    rules = """
    The game of REVERSE consists of a row of 9 squares, numbered 1 to 9.  The
    computer selects a random arrangement of the numbers and you try to get
    them back in order by repeatedly telling the computer how many numbers to
    reverse.  The computer complies and shows you the new arrangement.  You
    continue until you get them in order.  If you can do it in 6 moves, you're
    a genius.  7 or 8 moves is good.  9 moves is O.K.  10 moves is poor.  More
    than 10 moves and you're a dunce.
    """
    print(textwrap.fill(rules, width=72))
    # 使用 textwrap.dedent() 方法创建多行字符串，用于存储游戏规则说明
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
    # 打印游戏规则说明
    print(help)
    # 打印空行
    print()
# 如果当前模块被直接执行，则执行 main 函数
if __name__ == "__main__":
    # 尝试执行 main 函数
    try:
        main()
    # 捕获键盘中断异常，不做任何处理
    except KeyboardInterrupt:
        pass
```