# `34_Digits\python\Digits.py`

```
import random  # 导入 random 模块，用于生成随机数
from typing import List  # 从 typing 模块中导入 List 类型，用于类型提示


def print_intro() -> None:
    print("                                DIGITS")  # 打印游戏标题
    print("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 打印游戏信息
    print("\n\n")  # 打印空行
    print("THIS IS A GAME OF GUESSING.")  # 打印游戏说明


def read_instruction_choice() -> bool:
    print("FOR INSTRUCTIONS, TYPE '1', ELSE TYPE '0' ? ")  # 打印提示信息
    try:
        choice = int(input())  # 获取用户输入的选择
        return choice == 1  # 如果用户选择了1，则返回 True，否则返回 False
    except (ValueError, TypeError):  # 捕获可能的异常
        return False  # 如果出现异常，则返回 False
        # 循环直到输入有效的数字
        # 输入数字
            num = input("Enter a number (0, 1, or 2): ")
            # 检查输入是否为有效数字
            if num in ['0', '1', '2']:
                # 将有效数字添加到列表中
                numbers.append(int(num))
                # 设置有效输入为 True，退出循环
                valid_input = True
            else:
                # 输入无效，提示重新输入
                print("Invalid input. Please enter a number (0, 1, or 2).")

    # 返回输入的数字列表
    return numbers
            try:
                # 尝试将用户输入转换为整数
                n = int(input())
                # 如果成功转换，将 valid_input 标记为 True，并将数字添加到列表中
                valid_input = True
                numbers.append(n)
            except (TypeError, ValueError):
                # 如果转换失败，打印错误消息
                print("!NUMBER EXPECTED - RETRY INPUT LINE")

    # 返回数字列表
    return numbers


def read_continue_choice() -> bool:
    # 打印提示消息，询问用户是否要再次尝试
    print("\nDO YOU WANT TO TRY AGAIN (1 FOR YES, 0 FOR NO) ? ")
    try:
        # 尝试将用户输入转换为整数
        choice = int(input())
        # 返回用户输入是否为 1
        return choice == 1
    except (ValueError, TypeError):
        # 如果转换失败，返回 False
        return False


def print_summary_report(running_correct: int) -> None:
    print()  # 打印空行
    if running_correct > 10:  # 如果 running_correct 大于 10
        print()  # 打印空行
        print("I GUESSED MORE THAN 1/3 OF YOUR NUMBERS.")  # 打印消息
        print("I WIN.\u0007")  # 打印消息并发出响铃
    elif running_correct < 10:  # 如果 running_correct 小于 10
        print("I GUESSED LESS THAN 1/3 OF YOUR NUMBERS.")  # 打印消息
        print("YOU BEAT ME.  CONGRATULATIONS *****")  # 打印消息
    else:  # 否则
        print("I GUESSED EXACTLY 1/3 OF YOUR NUMBERS.")  # 打印消息
        print("IT'S A TIE GAME.")  # 打印消息


def main() -> None:
    print_intro()  # 调用打印介绍的函数
    if read_instruction_choice():  # 如果读取指令选择的函数返回 True
        print_instructions()  # 调用打印指令的函数

    a = 0  # 初始化变量 a 为 0
    b = 1  # 初始化变量 b 为 1
    c = 3  # 定义变量 c 并赋值为 3

    m = [[1] * 3 for _ in range(27)]  # 创建一个包含 27 个子列表的列表，每个子列表包含 3 个元素，值为 1
    k = [[9] * 3 for _ in range(3)]  # 创建一个包含 3 个子列表的列表，每个子列表包含 3 个元素，值为 9
    l = [[3] * 3 for _ in range(9)]  # 创建一个包含 9 个子列表的列表，每个子列表包含 3 个元素，值为 3

    continue_game = True  # 定义变量 continue_game 并赋值为 True
    while continue_game:  # 当 continue_game 为 True 时执行循环
        l[0][0] = 2  # 修改 l 列表中第一个子列表的第一个元素的值为 2
        l[4][1] = 2  # 修改 l 列表中第五个子列表的第二个元素的值为 2
        l[8][2] = 2  # 修改 l 列表中第九个子列表的第三个元素的值为 2
        z: float = 26  # 定义变量 z 并赋值为 26，类型为浮点数
        z1: float = 8  # 定义变量 z1 并赋值为 8，类型为浮点数
        z2 = 2  # 定义变量 z2 并赋值为 2
        running_correct = 0  # 定义变量 running_correct 并赋值为 0

        for _round in range(1, 4):  # 循环遍历范围为 1 到 3
            valid_numbers = False  # 定义变量 valid_numbers 并赋值为 False
            numbers = []  # 定义变量 numbers 并赋值为空列表
            while not valid_numbers:  # 当 valid_numbers 为 False 时执行循环
                # 打印空行
                print()
                # 调用read_10_numbers函数，将返回的数字存储在numbers变量中
                numbers = read_10_numbers()
                # 初始化valid_numbers为True
                valid_numbers = True
                # 遍历numbers列表中的每个数字
                for number in numbers:
                    # 如果数字小于0或大于2
                    if number < 0 or number > 2:
                        # 打印错误提示信息
                        print("ONLY USE THE DIGITS '0', '1', OR '2'.")
                        print("LET'S TRY AGAIN.")
                        # 将valid_numbers设置为False
                        valid_numbers = False
                        # 退出循环
                        break

            # 打印表头
            print(
                "\n%-14s%-14s%-14s%-14s"
                % ("MY GUESS", "YOUR NO.", "RESULT", "NO. RIGHT")
            )

            # 遍历numbers列表中的每个数字
            for number in numbers:
                # 初始化s为0
                s = 0
                # 初始化my_guess为0
                my_guess = 0
                # 遍历0到2的范围
                for j in range(0, 3):
                    # 原作者的意图是什么？
                    # 第一个表达式总是得到0，因为a始终为0
                    s1 = a * k[z2][j] + b * l[int(z1)][j] + c * m[int(z)][j]
                    # 如果s小于s1，则更新s和my_guess
                    if s < s1:
                        s = s1
                        my_guess = j
                    # 如果s1等于s且随机数大于等于0.5，则更新my_guess
                    elif s1 == s and random.random() >= 0.5:
                        my_guess = j

                result = ""

                # 如果my_guess不等于number，则结果为"WRONG"
                if my_guess != number:
                    result = "WRONG"
                # 否则，结果为"RIGHT"，并更新相关变量
                else:
                    running_correct += 1
                    result = "RIGHT"
                    m[int(z)][number] = m[int(z)][number] + 1
                    l[int(z1)][number] = l[int(z1)][number] + 1
                    k[int(z2)][number] = k[int(z2)][number] + 1
                    z = z - (z / 9) * 9
                    z = 3 * z + number
                print(
                    "\n%-14d%-14d%-14s%-14d"
                    % (my_guess, number, result, running_correct)
                )
                # 打印格式化后的数据，包括 my_guess, number, result, running_correct

                z1 = z - (z / 9) * 9
                # 计算 z1 的值，z 除以 9 的余数

                z2 = number
                # 将 z2 的值设置为 number

        print_summary_report(running_correct)
        # 调用打印总结报告的函数，传入 running_correct 参数

        continue_game = read_continue_choice()
        # 调用读取继续游戏选择的函数，并将结果赋值给 continue_game

    print("\nTHANKS FOR THE GAME.")
    # 打印感谢信息

if __name__ == "__main__":
    main()
    # 如果当前文件被直接运行，则调用 main 函数
```