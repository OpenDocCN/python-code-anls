# `basic-computer-games\36_Flip_Flop\python\flipflop.py`

```py
# Flip Flop
#
# The object of this game is to change a row of ten X's
# X X X X X X X X X X
# to a row of ten O's:
# O O O O O O O O O O
# by typing in a number corresponding
# to the position of an "X" in the line. On
# some numbers one position will
# change while on other numbers, two
# will change. For example, inputting a 3
# may reverse the X and O in position 3,
# but it might possibly reverse some
# other position too! You ought to be able
# to change all 10 in 12 or fewer
# moves. Can you figure out a good win-
# ning strategy?
# To reset the line to all X's (same
# game), type 0 (zero). To start a new
# game at any point, type 11.
# The original author of this game was
# Michael Kass of New Hyde Park, New
# York.
import math
import random
from typing import Callable, List, Tuple

# 字典，用于将 X 转换为 O，O 转换为 X
flip_dict = {"X": "O", "O": "X"}


def flip_bits(
    row: List[str], m: int, n: int, r_function: Callable[[int], float]
) -> Tuple[List[str], int]:
    """
    Function that flips the positions at the computed steps
    """
    # 当 m 和 n 相等时，执行循环
    while m == n:
        # 计算 r
        r = r_function(n)
        # 计算新的 n
        n_tmp = r - int(math.floor(r))
        n = int(10 * n_tmp)
        # 如果 row[n] 是 X，则将其改为 O，并跳出循环
        if row[n] == "X":
            row[n] = "O"
            break
        # 如果 row[n] 是 O，则将其改为 X，并跳出循环
        elif row[n] == "O":
            row[n] = "X"
    return row, n


def print_instructions() -> None:
    # 打印游戏说明
    print(" " * 32 + "FLIPFLOP")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n" * 2)
    print("THE OBJECT OF THIS PUZZLE IS TO CHANGE THIS:\n")
    print("X X X X X X X X X X\n")
    print("TO THIS:\n")
    print("O O O O O O O O O O\n")
    print("BY TYPING TH NUMBER CORRESPONDING TO THE POSITION OF THE")
    print("LETTER ON SOME NUMBERS, ONE POSITION WILL CHANGE, ON")
    print("OTHERS, TWO WILL CHANGE. TO RESET LINE TO ALL X'S, TYPE 0")
    print("(ZERO) AND TO START OVER IN THE MIDDLE OF A GAME, TYPE ")
    print("11 (ELEVEN).\n")


def main() -> None:
    # 生成一个随机数
    q = random.random()

    # 打印游戏开始时的初始行
    print("HERE IS THE STARTING LINE OF X'S.\n")
    # 创建一个包含11个空字符串和10个"X"的列表，用于表示游戏的一行
    # 初始化计数器为0
    # 初始化n为-1
    # 初始化合法移动标志为True
    row = [""] + ["X"] * 10
    counter_turns = 0
    n = -1
    legal_move = True
    # 当第1到第10个元素不全为"O"时，循环执行以下操作
    while row[1:] != ["O"] * 10:
        # 如果是合法移动，则打印1到10的数字
        # 打印当前行的状态
        if legal_move:
            print(" ".join([str(i) for i in range(1, 11)]))
            print(" ".join(row[1:]) + "\n")
        # 从用户输入获取一个字符串
        m_str = input("INPUT THE NUMBER\n")
        try:
            # 将输入的字符串转换为整数
            m = int(m_str)
            # 如果m大于11或小于0，则引发ValueError
            if m > 11 or m < 0:
                raise ValueError()
        except ValueError:
            # 如果出现异常，则打印错误信息，将合法移动标志设置为False，并继续下一次循环
            print("ILLEGAL ENTRY--TRY AGAIN")
            legal_move = False
            continue
        # 如果m等于11，则完全重置谜题
        # 重置计数器为0
        # 重置行为包含一个空字符串和10个"X"
        # 生成一个随机数q
        elif m == 11:
            counter_turns = 0
            row = [""] + ["X"] * 10
            q = random.random()
            continue
        # 如果m等于0，则重置行为包含一个空字符串和10个"X"
        elif m == 0:
            row = [""] + ["X"] * 10
        # 如果m等于n，则翻转row[n]的值，并调用flip_bits函数
        else:
            n = m
            row[n] = flip_dict[row[n]]
            r_function = lambda n_t: (
                math.tan(q + n_t / q - n_t)
                - math.sin(n_t * 2 + q)
                + 336 * math.sin(8 * n_t)
            )
            row, n = flip_bits(row, m, n, r_function)
    
        # 计数器加1
        counter_turns += 1
        print()
    
    # 如果计数器小于等于12，则打印猜对的信息和猜测次数
    # 否则打印猜错的信息和猜测次数
    return
# 如果当前模块是主程序，则执行以下代码
if __name__ == "__main__":
    # 打印说明
    print_instructions()

    # 初始化变量 another 为空字符串
    another = ""
    # 当用户输入不等于 "NO" 时，循环执行以下代码
    while another != "NO":
        # 调用主函数
        main()
        # 用户输入是否想尝试另一个谜题
        another = input("DO YOU WANT TO TRY ANOTHER PUZZLE\n")
```