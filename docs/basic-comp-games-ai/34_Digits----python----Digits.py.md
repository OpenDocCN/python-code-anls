# `basic-computer-games\34_Digits\python\Digits.py`

```

# 导入 random 模块和 List 类型
import random
from typing import List


# 打印游戏介绍
def print_intro() -> None:
    print("                                DIGITS")
    print("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n\n")
    print("THIS IS A GAME OF GUESSING.")


# 读取用户是否需要游戏说明的选择
def read_instruction_choice() -> bool:
    print("FOR INSTRUCTIONS, TYPE '1', ELSE TYPE '0' ? ")
    try:
        choice = int(input())
        return choice == 1
    except (ValueError, TypeError):
        return False


# 打印游戏说明
def print_instructions() -> None:
    # 打印游戏说明
    print("\n")
    print("PLEASE TAKE A PIECE OF PAPER AND WRITE DOWN")
    # 省略部分说明内容
    print("THIRTY TIMES AT RANDOM.")
    print()


# 读取用户输入的十个数字
def read_10_numbers() -> List[int]:
    print("TEN NUMBERS, PLEASE ? ")
    numbers = []

    for _ in range(10):
        valid_input = False
        while not valid_input:
            try:
                n = int(input())
                valid_input = True
                numbers.append(n)
            except (TypeError, ValueError):
                print("!NUMBER EXPECTED - RETRY INPUT LINE")

    return numbers


# 读取用户是否继续游戏的选择
def read_continue_choice() -> bool:
    print("\nDO YOU WANT TO TRY AGAIN (1 FOR YES, 0 FOR NO) ? ")
    try:
        choice = int(input())
        return choice == 1
    except (ValueError, TypeError):
        return False


# 打印游戏总结报告
def print_summary_report(running_correct: int) -> None:
    print()
    # 根据猜对的数量打印不同的总结报告
    if running_correct > 10:
        print()
        print("I GUESSED MORE THAN 1/3 OF YOUR NUMBERS.")
        print("I WIN.\u0007")
    elif running_correct < 10:
        print("I GUESSED LESS THAN 1/3 OF YOUR NUMBERS.")
        print("YOU BEAT ME.  CONGRATULATIONS *****")
    else:
        print("I GUESSED EXACTLY 1/3 OF YOUR NUMBERS.")
        print("IT'S A TIE GAME.")


# 游戏主函数
def main() -> None:
    # 打印游戏介绍
    print_intro()
    # 如果用户需要游戏说明，则打印游戏说明
    if read_instruction_choice():
        print_instructions()

    # 初始化一些变量
    a = 0
    b = 1
    c = 3
    m = [[1] * 3 for _ in range(27)]
    k = [[9] * 3 for _ in range(3)]
    l = [[3] * 3 for _ in range(9)]  # noqa: E741

    continue_game = True
    while continue_game:
        # 省略部分初始化和计算逻辑
        running_correct = 0

        for _round in range(1, 4):
            valid_numbers = False
            numbers = []
            while not valid_numbers:
                print()
                numbers = read_10_numbers()
                valid_numbers = True
                for number in numbers:
                    if number < 0 or number > 2:
                        print("ONLY USE THE DIGITS '0', '1', OR '2'.")
                        print("LET'S TRY AGAIN.")
                        valid_numbers = False
                        break

            print(
                "\n%-14s%-14s%-14s%-14s"
                % ("MY GUESS", "YOUR NO.", "RESULT", "NO. RIGHT")
            )

            for number in numbers:
                s = 0
                my_guess = 0
                for j in range(0, 3):
                    # 省略部分逻辑
                    if s < s1:
                        s = s1
                        my_guess = j
                    elif s1 == s and random.random() >= 0.5:
                        my_guess = j

                result = ""

                if my_guess != number:
                    result = "WRONG"
                else:
                    running_correct += 1
                    result = "RIGHT"
                    # 省略部分逻辑
                print(
                    "\n%-14d%-14d%-14s%-14d"
                    % (my_guess, number, result, running_correct)
                )

                # 省略部分逻辑

        print_summary_report(running_correct)
        continue_game = read_continue_choice()

    print("\nTHANKS FOR THE GAME.")


if __name__ == "__main__":
    main()

```