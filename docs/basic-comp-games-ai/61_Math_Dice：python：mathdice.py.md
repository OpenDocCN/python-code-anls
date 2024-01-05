# `d:/src/tocomm/basic-computer-games\61_Math_Dice\python\mathdice.py`

```
from random import randint  # 从 random 模块中导入 randint 函数

print("Math Dice")  # 打印 "Math Dice"
print("https://github.com/coding-horror/basic-computer-games")  # 打印 GitHub 仓库链接
print()  # 打印空行
print(
    """This program generates images of two dice.
When two dice and an equals sign followed by a question
mark have been printed, type your answer, and hit the ENTER
key.
To conclude the program, type 0.
"""
)  # 打印程序说明文本


def print_dice(n: int) -> None:  # 定义一个函数 print_dice，参数为整数 n，返回类型为 None
    def print_0() -> None:  # 定义一个内部函数 print_0，返回类型为 None
        print("|     |")  # 打印一个表示数字 0 的图案

    def print_2() -> None:  # 定义一个内部函数 print_2，返回类型为 None
        print("| * * |")  # 打印包含星号的特定格式的字符串

    print(" ----- ")  # 打印一条横线

    if n in [4, 5, 6]:  # 如果 n 在列表 [4, 5, 6] 中
        print_2()  # 调用函数 print_2()
    elif n in [2, 3]:  # 如果 n 在列表 [2, 3] 中
        print("| *   |")  # 打印包含星号的特定格式的字符串
    else:  # 否则
        print_0()  # 调用函数 print_0()

    if n in [1, 3, 5]:  # 如果 n 在列表 [1, 3, 5] 中
        print("|  *  |")  # 打印包含星号的特定格式的字符串
    elif n in [2, 4]:  # 如果 n 在列表 [2, 4] 中
        print_0()  # 调用函数 print_0()
    else:  # 否则
        print_2()  # 调用函数 print_2()

    if n in [4, 5, 6]:  # 如果 n 在列表 [4, 5, 6] 中
        print_2()  # 调用函数 print_2()
    elif n in [2, 3]:  # 如果 n 的值在列表 [2, 3] 中
        print("|   * |")  # 打印出 "|   * |"
    else:  # 否则
        print_0()  # 调用函数 print_0()

    print(" ----- ")  # 打印出 " ----- "


def main() -> None:  # 定义一个名为 main 的函数，不返回任何值

    while True:  # 无限循环
        d1 = randint(1, 6)  # 生成一个 1 到 6 之间的随机整数，赋值给变量 d1
        d2 = randint(1, 6)  # 生成一个 1 到 6 之间的随机整数，赋值给变量 d2
        guess = 13  # 将变量 guess 的值设为 13

        print_dice(d1)  # 调用函数 print_dice()，传入参数 d1
        print("   +")  # 打印出 "   +"
        print_dice(d2)  # 调用函数 print_dice()，传入参数 d2
        print("   =")  # 打印出 "   ="
        tries = 0  # 初始化尝试次数为0
        while guess != (d1 + d2) and tries < 2:  # 当猜测值不等于两个骰子点数之和且尝试次数小于2时循环
            if tries == 1:  # 如果尝试次数为1
                print("No, count the spots and give another answer.")  # 输出提示信息
            try:  # 尝试执行以下代码
                guess = int(input())  # 获取用户输入的整数值作为猜测值
            except ValueError:  # 如果出现值错误异常
                print("That's not a number!")  # 输出提示信息
            if guess == 0:  # 如果猜测值为0
                exit()  # 退出程序
            tries += 1  # 尝试次数加1

        if guess != (d1 + d2):  # 如果猜测值不等于两个骰子点数之和
            print(f"No, the answer is {d1 + d2}!")  # 输出提示信息，显示正确答案
        else:  # 否则
            print("Correct!")  # 输出提示信息，显示回答正确

        print("The dice roll again....")  # 输出提示信息，显示骰子再次滚动
# 如果当前脚本被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```

这段代码用于判断当前脚本是否被直接执行。如果当前脚本被直接执行，则调用 main() 函数。这样可以确保在作为模块导入时不会执行 main() 函数，只有在作为独立脚本执行时才会执行 main() 函数。
```