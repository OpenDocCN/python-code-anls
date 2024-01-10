# `basic-computer-games\61_Math_Dice\python\mathdice.py`

```
# 从 random 模块中导入 randint 函数
from random import randint

# 打印标题和来源信息
print("Math Dice")
print("https://github.com/coding-horror/basic-computer-games")
print()

# 打印游戏规则说明
print(
    """This program generates images of two dice.
When two dice and an equals sign followed by a question
mark have been printed, type your answer, and hit the ENTER
key.
To conclude the program, type 0.
"""
)

# 定义打印骰子图像的函数
def print_dice(n: int) -> None:
    # 定义打印空骰子的函数
    def print_0() -> None:
        print("|     |")

    # 定义打印两点骰子的函数
    def print_2() -> None:
        print("| * * |")

    # 打印骰子图像的上边框
    print(" ----- ")

    # 根据骰子点数打印不同的图像
    if n in [4, 5, 6]:
        print_2()
    elif n in [2, 3]:
        print("| *   |")
    else:
        print_0()

    if n in [1, 3, 5]:
        print("|  *  |")
    elif n in [2, 4]:
        print_0()
    else:
        print_2()

    if n in [4, 5, 6]:
        print_2()
    elif n in [2, 3]:
        print("|   * |")
    else:
        print_0()

    # 打印骰子图像的下边框
    print(" ----- ")

# 定义主函数
def main() -> None:

    # 无限循环，直到用户输入 0 结束程序
    while True:
        # 随机生成两个骰子的点数
        d1 = randint(1, 6)
        d2 = randint(1, 6)
        guess = 13

        # 打印第一个骰子的图像
        print_dice(d1)
        print("   +")
        # 打印加号
        print_dice(d2)
        print("   =")

        tries = 0
        # 循环直到用户猜对或者尝试次数超过2次
        while guess != (d1 + d2) and tries < 2:
            if tries == 1:
                print("No, count the spots and give another answer.")
            try:
                # 获取用户输入的猜测值
                guess = int(input())
            except ValueError:
                print("That's not a number!")
            # 如果用户输入 0，则退出程序
            if guess == 0:
                exit()
            tries += 1

        # 根据用户猜测的结果给出相应的提示
        if guess != (d1 + d2):
            print(f"No, the answer is {d1 + d2}!")
        else:
            print("Correct!")

        print("The dice roll again....")

# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```