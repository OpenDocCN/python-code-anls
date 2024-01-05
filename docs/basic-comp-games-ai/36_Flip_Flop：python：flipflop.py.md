# `d:/src/tocomm/basic-computer-games\36_Flip_Flop\python\flipflop.py`

```
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
# 导入 math 模块
import math
# 导入 random 模块
import random
# 导入 typing 模块中的 Callable、List 和 Tuple 类型
from typing import Callable, List, Tuple

# 定义一个字典，用于存储翻转操作的映射关系
flip_dict = {"X": "O", "O": "X"}

# 定义一个函数，用于执行翻转操作
def flip_bits(
    row: List[str], m: int, n: int, r_function: Callable[[int], float]
) -> Tuple[List[str], int]:
    """
    Function that flips the positions at the computed steps
    """
    # 当 m 等于 n 时，执行循环
    while m == n:
        # 使用 r_function 函数计算得到 r 值
        r = r_function(n)
        # 计算 n_tmp 值
        n_tmp = r - int(math.floor(r))
        # 更新 n 值
        n = int(10 * n_tmp)
        if row[n] == "X":  # 如果列表中的元素等于 "X"
            row[n] = "O"  # 将列表中的元素改为 "O"
            break  # 跳出循环
        elif row[n] == "O":  # 如果列表中的元素等于 "O"
            row[n] = "X"  # 将列表中的元素改为 "X"
    return row, n  # 返回修改后的列表和索引值


def print_instructions() -> None:  # 定义一个打印说明的函数，不返回任何值
    print(" " * 32 + "FLIPFLOP")  # 打印标题
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 打印创意计算的信息
    print("\n" * 2)  # 打印两个空行
    print("THE OBJECT OF THIS PUZZLE IS TO CHANGE THIS:\n")  # 打印游戏目标
    print("X X X X X X X X X X\n")  # 打印初始状态
    print("TO THIS:\n")  # 打印目标状态
    print("O O O O O O O O O O\n")  # 打印目标状态
    print("BY TYPING TH NUMBER CORRESPONDING TO THE POSITION OF THE")  # 打印游戏说明
    print("LETTER ON SOME NUMBERS, ONE POSITION WILL CHANGE, ON")  # 打印游戏说明
    print("OTHERS, TWO WILL CHANGE. TO RESET LINE TO ALL X'S, TYPE 0")  # 打印游戏说明
    print("(ZERO) AND TO START OVER IN THE MIDDLE OF A GAME, TYPE ")  # 打印游戏说明
    print("11 (ELEVEN).\n")  # 打印字符串 "11 (ELEVEN).\n"

def main() -> None:
    q = random.random()  # 生成一个随机数并赋值给变量 q

    print("HERE IS THE STARTING LINE OF X'S.\n")  # 打印字符串 "HERE IS THE STARTING LINE OF X'S.\n"
    # We add an extra 0-th item because this sometimes is set to something
    # but we never check what it is for completion of the puzzle
    row = [""] + ["X"] * 10  # 创建一个包含 11 个元素的列表，第一个元素为空字符串，其余元素为 "X"
    counter_turns = 0  # 初始化变量 counter_turns 为 0
    n = -1  # 初始化变量 n 为 -1
    legal_move = True  # 初始化变量 legal_move 为 True
    while row[1:] != ["O"] * 10:  # 当列表 row 的第二个元素到最后一个元素不全为 "O" 时执行循环
        if legal_move:  # 如果 legal_move 为 True
            print(" ".join([str(i) for i in range(1, 11)]))  # 打印从 1 到 10 的数字并以空格分隔
            print(" ".join(row[1:]) + "\n")  # 打印列表 row 的第二个元素到最后一个元素并以空格分隔，末尾换行
        m_str = input("INPUT THE NUMBER\n")  # 获取用户输入的字符串并赋值给变量 m_str
        try:  # 尝试执行以下代码
            m = int(m_str)  # 将用户输入的字符串转换为整数并赋值给变量 m
            if m > 11 or m < 0:  # 如果 m 大于 11 或者小于 0
                raise ValueError()  # 抛出值错误异常
        except ValueError:  # 捕获值错误异常
            print("ILLEGAL ENTRY--TRY AGAIN")  # 打印非法输入提示
            legal_move = False  # 将 legal_move 设置为 False
            continue  # 继续下一次循环
        legal_move = True  # 将 legal_move 设置为 True
        if m == 11:  # 如果 m 等于 11
            # completely reset the puzzle
            counter_turns = 0  # 将 counter_turns 设置为 0
            row = [""] + ["X"] * 10  # 重置 row 列表
            q = random.random()  # 重新生成随机数 q
            continue  # 继续下一次循环
        elif m == 0:  # 如果 m 等于 0
            # reset the board, but not the counter or the random number
            row = [""] + ["X"] * 10  # 重置 row 列表
        elif m == n:  # 如果 m 等于 n
            row[n] = flip_dict[row[n]]  # 将 row[n] 的值翻转
            r_function = lambda n_t: 0.592 * (1 / math.tan(q / n_t + q)) / math.sin(n_t * 2 + q)  # 设置 r_function 的值为 lambda 表达式
            ) - math.cos(n_t)  # 计算新的 n_t 值
            row, n = flip_bits(row, m, n, r_function)  # 调用 flip_bits 函数，更新 row 和 n 的值
        else:
            n = m  # 将 n 的值更新为 m
            row[n] = flip_dict[row[n]]  # 使用 flip_dict 更新 row[n] 的值
            r_function = lambda n_t: (  # 定义新的 r_function
                math.tan(q + n_t / q - n_t)  # 计算 tan 值
                - math.sin(n_t * 2 + q)  # 计算 sin 值
                + 336 * math.sin(8 * n_t)  # 计算 sin 值
            )
            row, n = flip_bits(row, m, n, r_function)  # 调用 flip_bits 函数，更新 row 和 n 的值

        counter_turns += 1  # 计数器加一
        print()  # 打印空行

    if counter_turns <= 12:  # 判断猜测次数是否小于等于 12
        print(f"VERY GOOD. YOU GUESSED IT IN ONLY {counter_turns} GUESSES.")  # 打印猜测次数
    else:
        print(f"TRY HARDER NEXT TIME. IT TOOK YOU {counter_turns} GUESSES.")  # 打印猜测次数
    return  # 返回
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 打印说明
    print_instructions()

    # 初始化变量 another 为空字符串
    another = ""
    # 当用户输入不等于 "NO" 时，循环执行 main() 函数
    while another != "NO":
        main()
        # 用户输入是否想尝试另一个谜题
        another = input("DO YOU WANT TO TRY ANOTHER PUZZLE\n")
```