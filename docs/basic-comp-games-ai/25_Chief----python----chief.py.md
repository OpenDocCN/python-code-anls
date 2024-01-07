# `basic-computer-games\25_Chief\python\chief.py`

```

# 打印一个闪电形状
def print_lightning_bolt() -> None:
    # 打印一行星号
    print("*" * 36)
    # 初始化变量n为24
    n = 24
    # 循环打印闪电形状的上半部分
    while n > 16:
        print(" " * n + "x x")
        n -= 1
    # 打印闪电形状的中间部分
    print(" " * 16 + "x xxx")
    print(" " * 15 + "x   x")
    print(" " * 14 + "xxx x")
    n -= 1
    # 循环打印闪电形状的下半部分
    while n > 8:
        print(" " * n + "x x")
        n -= 1
    print(" " * 8 + "xx")
    print(" " * 7 + "x")
    # 打印一行星号
    print("*" * 36)


# 打印数学解题过程
def print_solution(n: float) -> None:
    # 打印数学解题过程的每一步
    print(f"\n{n} plus 3 gives {n + 3}. This Divided by 5 equals {(n + 3) / 5}")
    print(f"This times 8 gives {((n + 3) / 5) * 8}. If we divide 5 and add 5.")
    print(
        f"We get {(((n + 3) / 5) * 8) / 5 + 5}, "
        f"which, minus 1 equals {((((n + 3) / 5) * 8) / 5 + 5) - 1}"
    )


# 数学游戏
def game() -> None:
    # 打印游戏规则
    print("\nTake a Number and ADD 3. Now, Divide this number by 5 and")
    print("multiply by 8. Now, Divide by 5 and add the same. Subtract 1")

    # 用户输入一个数
    you_have = float(input("\nWhat do you have? "))
    # 计算计算机猜测的数
    comp_guess = (((you_have - 4) * 5) / 8) * 5 - 3
    # 用户确认计算机猜测是否正确
    first_guess_right = input(
        f"\nI bet your number was {comp_guess} was I right(Yes or No)? "
    )

    # 根据用户的回答进行不同的处理
    if first_guess_right.lower() == "yes":
        # 用户确认计算机猜测正确
        print("\nHuh, I Knew I was unbeatable")
        print("And here is how i did it")
        # 打印数学解题过程
        print_solution(comp_guess)
        input()
    else:
        # 用户确认计算机猜测错误，继续游戏
        original_number = float(input("\nHUH!! what was you original number? "))

        if original_number == comp_guess:
            # 用户确认计算机猜测正确
            print("\nThat was my guess, AHA i was right")
            print(
                "Shamed to accept defeat i guess, don't worry you can master mathematics too"
            )
            print("Here is how i did it")
            # 打印数学解题过程
            print_solution(comp_guess)
            input()
        else:
            # 用户继续否认计算机猜测，计算机展示数学解题过程
            print("\nSo you think you're so smart, EH?")
            print("Now, Watch")
            print_solution(original_number)

            # 用户确认是否相信计算机的解题过程
            believe_me = input("\nNow do you believe me? ")

            if believe_me.lower() == "yes":
                # 用户确认相信计算机的解题过程
                print("\nOk, Lets play again sometime bye!!!!")
                input()
            else:
                # 用户继续否认计算机的解题过程，计算机展示闪电形状
                print("\nYOU HAVE MADE ME VERY MAD!!!!!")
                print("BY THE WRATH OF THE MATHEMATICS AND THE RAGE OF THE GODS")
                print("THERE SHALL BE LIGHTNING!!!!!!!")
                # 打印闪电形状
                print_lightning_bolt()
                print("\nI Hope you believe me now, for your own sake")
                input()


if __name__ == "__main__":
    # 主程序入口，询问用户是否准备好进行游戏
    print("I am CHIEF NUMBERS FREEK, The GREAT INDIAN MATH GOD.")
    play = input("\nAre you ready to take the test you called me out for(Yes or No)? ")
    if play.lower() == "yes":
        # 用户确认准备好进行游戏，开始游戏
        game()
    else:
        # 用户不准备进行游戏，结束程序
        print("Ok, Nevermind. Let me go back to my great slumber, Bye")
        input()

```