# `basic-computer-games\25_Chief\python\chief.py`

```
# 打印闪电形状
def print_lightning_bolt() -> None:
    # 打印一行星号
    print("*" * 36)
    # 初始化变量 n 为 24
    n = 24
    # 循环打印空格和"x x"，直到 n 小于等于 16
    while n > 16:
        print(" " * n + "x x")
        n -= 1
    # 打印"x xxx"
    print(" " * 16 + "x xxx")
    # 打印"x   x"
    print(" " * 15 + "x   x")
    # 打印"xxx x"
    print(" " * 14 + "xxx x")
    # 继续减小 n，循环打印空格和"x x"，直到 n 小于等于 8
    n -= 1
    while n > 8:
        print(" " * n + "x x")
        n -= 1
    # 打印"xx"
    print(" " * 8 + "xx")
    # 打印"x"
    print(" " * 7 + "x")
    # 打印一行星号
    print("*" * 36)


# 打印解决方案
def print_solution(n: float) -> None:
    # 打印计算结果
    print(f"\n{n} plus 3 gives {n + 3}. This Divided by 5 equals {(n + 3) / 5}")
    print(f"This times 8 gives {((n + 3) / 5) * 8}. If we divide 5 and add 5.")
    print(
        f"We get {(((n + 3) / 5) * 8) / 5 + 5}, "
        f"which, minus 1 equals {((((n + 3) / 5) * 8) / 5 + 5) - 1}"
    )


# 游戏函数
def game() -> None:
    # 打印游戏提示
    print("\nTake a Number and ADD 3. Now, Divide this number by 5 and")
    print("multiply by 8. Now, Divide by 5 and add the same. Subtract 1")

    # 获取用户输入的数字
    you_have = float(input("\nWhat do you have? "))
    # 计算计算机猜测的数字
    comp_guess = (((you_have - 4) * 5) / 8) * 5 - 3
    # 获取用户对计算机猜测是否正确的回答
    first_guess_right = input(
        f"\nI bet your number was {comp_guess} was I right(Yes or No)? "
    )

    # 判断用户回答是否正确
    if first_guess_right.lower() == "yes":
        # 打印计算机的表现和解决方案
        print("\nHuh, I Knew I was unbeatable")
        print("And here is how i did it")
        print_solution(comp_guess)
        input()
    # 如果猜测错误，要求用户输入原始数字
    else:
        original_number = float(input("\nHUH!! what was you original number? "))

        # 如果用户输入的原始数字和计算的结果相等
        if original_number == comp_guess:
            # 打印猜测正确的消息
            print("\nThat was my guess, AHA i was right")
            print(
                "Shamed to accept defeat i guess, don't worry you can master mathematics too"
            )
            print("Here is how i did it")
            # 打印计算过程
            print_solution(comp_guess)
            input()
        else:
            # 打印猜测错误的消息
            print("\nSo you think you're so smart, EH?")
            print("Now, Watch")
            # 打印原始数字的计算过程
            print_solution(original_number)

            # 要求用户相信计算结果
            believe_me = input("\nNow do you believe me? ")

            # 如果用户相信计算结果
            if believe_me.lower() == "yes":
                # 打印结束游戏的消息
                print("\nOk, Lets play again sometime bye!!!!")
                input()
            else:
                # 打印用户不相信的消息
                print("\nYOU HAVE MADE ME VERY MAD!!!!!")
                print("BY THE WRATH OF THE MATHEMATICS AND THE RAGE OF THE GODS")
                print("THERE SHALL BE LIGHTNING!!!!!!!")
                # 打印闪电效果
                print_lightning_bolt()
                print("\nI Hope you believe me now, for your own sake")
                input()
# 如果当前模块被直接执行
if __name__ == "__main__":
    # 打印欢迎信息
    print("I am CHIEF NUMBERS FREEK, The GREAT INDIAN MATH GOD.")
    # 获取用户输入，询问是否准备好进行测试
    play = input("\nAre you ready to take the test you called me out for(Yes or No)? ")
    # 如果用户输入的是yes（不区分大小写）
    if play.lower() == "yes":
        # 调用game函数开始游戏
        game()
    # 如果用户输入的不是yes
    else:
        # 打印消息，表示放弃测试
        print("Ok, Nevermind. Let me go back to my great slumber, Bye")
        # 等待用户输入，然后退出程序
        input()
```