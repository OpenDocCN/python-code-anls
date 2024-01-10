# `basic-computer-games\34_Digits\python\Digits.py`

```
# 导入 random 模块
import random
# 导入 List 类型
from typing import List

# 打印游戏介绍
def print_intro() -> None:
    print("                                DIGITS")
    print("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n\n")
    print("THIS IS A GAME OF GUESSING.")

# 读取用户是否需要查看游戏说明
def read_instruction_choice() -> bool:
    print("FOR INSTRUCTIONS, TYPE '1', ELSE TYPE '0' ? ")
    try:
        choice = int(input())
        return choice == 1
    except (ValueError, TypeError):
        return False

# 打印游戏说明
def print_instructions() -> None:
    print("\n")
    print("PLEASE TAKE A PIECE OF PAPER AND WRITE DOWN")
    print("THE DIGITS '0', '1', OR '2' THIRTY TIMES AT RANDOM.")
    print("ARRANGE THEM IN THREE LINES OF TEN DIGITS EACH.")
    print("I WILL ASK FOR THEN TEN AT A TIME.")
    print("I WILL ALWAYS GUESS THEM FIRST AND THEN LOOK AT YOUR")
    print("NEXT NUMBER TO SEE IF I WAS RIGHT. BY PURE LUCK,")
    print("I OUGHT TO BE RIGHT TEN TIMES. BUT I HOPE TO DO BETTER")
    print("THAN THAT *****")
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

# 读取用户是否继续游戏
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
    if running_correct > 10:
        print()
        print("I GUESSED MORE THAN 1/3 OF YOUR NUMBERS.")
        print("I WIN.\u0007")
    elif running_correct < 10:
        print("I GUESSED LESS THAN 1/3 OF YOUR NUMBERS.")
        print("YOU BEAT ME.  CONGRATULATIONS *****")
    # 如果不满足以上两个条件，则执行以下代码
    # 打印“我猜对了你的数字的三分之一”
    print("I GUESSED EXACTLY 1/3 OF YOUR NUMBERS.")
    # 打印“这是一场平局游戏”
    print("IT'S A TIE GAME.")
# 定义主函数，不返回任何结果
def main() -> None:
    # 打印游戏介绍
    print_intro()
    # 读取用户选择的指令，如果为真则打印游戏指令
    if read_instruction_choice():
        print_instructions()

    # 初始化变量 a, b, c
    a = 0
    b = 1
    c = 3

    # 创建一个 27x3 的二维数组，每个元素为 1
    m = [[1] * 3 for _ in range(27)]
    # 创建一个 3x3 的二维数组，每个元素为 9
    k = [[9] * 3 for _ in range(3)]
    # 创建一个 9x3 的二维数组，每个元素为 3
    l = [[3] * 3 for _ in range(9)]  # noqa: E741

    # 设置继续游戏标志为真
    continue_game = True
    # 打印感谢信息
    print("\nTHANKS FOR THE GAME.")


# 如果当前脚本为主程序，则执行主函数
if __name__ == "__main__":
    main()
```