# `d:/src/tocomm/basic-computer-games\25_Chief\python\chief.py`

```
def print_lightning_bolt() -> None:
    # 打印一个由 "*" 组成的长度为 36 的闪电形状
    print("*" * 36)
    # 初始化变量 n 为 24
    n = 24
    # 当 n 大于 16 时，循环打印空格和 "x x"，并将 n 减 1
    while n > 16:
        print(" " * n + "x x")
        n -= 1
    # 打印 "x xxx"，并将 n 减 1
    print(" " * 16 + "x xxx")
    # 打印 "x   x"
    print(" " * 15 + "x   x")
    # 打印 "xxx x"
    print(" " * 14 + "xxx x")
    # 将 n 减 1
    n -= 1
    # 当 n 大于 8 时，循环打印空格和 "x x"，并将 n 减 1
    while n > 8:
        print(" " * n + "x x")
        n -= 1
    # 打印 "xx"
    print(" " * 8 + "xx")
    # 打印 "x"
    print(" " * 7 + "x")
    # 打印一个由 "*" 组成的长度为 36 的闪电形状
    print("*" * 36)


def print_solution(n: float) -> None:
    # 打印给定数值加 3 的结果
    print(f"\n{n} plus 3 gives {n + 3}. This Divided by 5 equals {(n + 3) / 5}")
    print(f"This times 8 gives {((n + 3) / 5) * 8}. If we divide 5 and add 5.")
    # 打印出一个数乘以8的结果，这个数是(n + 3) / 5，然后再除以5并加上5

    print(
        f"We get {(((n + 3) / 5) * 8) / 5 + 5}, "
        f"which, minus 1 equals {((((n + 3) / 5) * 8) / 5 + 5) - 1}"
    )
    # 打印出一个数乘以8的结果，这个数是(n + 3) / 5，然后再除以5并加上5，然后再减去1


def game() -> None:
    print("\nTake a Number and ADD 3. Now, Divide this number by 5 and")
    print("multiply by 8. Now, Divide by 5 and add the same. Subtract 1")
    # 打印出游戏的规则

    you_have = float(input("\nWhat do you have? "))
    # 获取用户输入的数字，并转换成浮点数
    comp_guess = (((you_have - 4) * 5) / 8) * 5 - 3
    # 计算出计算机猜测的结果

    first_guess_right = input(
        f"\nI bet your number was {comp_guess} was I right(Yes or No)? "
    )
    # 获取用户对计算机猜测是否正确的回答

    if first_guess_right.lower() == "yes":
        print("\nHuh, I Knew I was unbeatable")
        print("And here is how i did it")
        # 如果用户回答是"yes"，则打印出计算机的胜利信息
        print_solution(comp_guess)  # 调用名为print_solution的函数，传入参数comp_guess，并打印函数返回的结果
        input()  # 等待用户输入任意内容，暂停程序的执行
    else:  # 如果前面的条件不成立
        original_number = float(input("\nHUH!! what was you original number? "))  # 从用户输入中获取原始数字，并将其转换为浮点数类型

        if original_number == comp_guess:  # 如果原始数字等于计算出的猜测值
            print("\nThat was my guess, AHA i was right")  # 打印消息表明程序猜对了
            print(
                "Shamed to accept defeat i guess, don't worry you can master mathematics too"
            )  # 打印鼓励的消息
            print("Here is how i did it")  # 打印消息表明程序将展示如何计算出猜测值
            print_solution(comp_guess)  # 调用名为print_solution的函数，传入参数comp_guess，并打印函数返回的结果
            input()  # 等待用户输入任意内容，暂停程序的执行
        else:  # 如果前面的条件不成立
            print("\nSo you think you're so smart, EH?")  # 打印挑衅的消息
            print("Now, Watch")  # 打印提示消息
            print_solution(original_number)  # 调用名为print_solution的函数，传入参数original_number，并打印函数返回的结果

            believe_me = input("\nNow do you believe me? ")  # 从用户输入中获取信任程度的回答
# 如果用户输入的是 "yes"，则打印消息并等待用户输入
if believe_me.lower() == "yes":
    print("\nOk, Lets play again sometime bye!!!!")
    input()
# 如果用户输入的不是 "yes"，则打印消息并执行一系列操作
else:
    print("\nYOU HAVE MADE ME VERY MAD!!!!!")
    print("BY THE WRATH OF THE MATHEMATICS AND THE RAGE OF THE GODS")
    print("THERE SHALL BE LIGHTNING!!!!!!!")
    print_lightning_bolt()
    print("\nI Hope you believe me now, for your own sake")
    input()

# 如果当前脚本被直接执行，则打印消息并等待用户输入
if __name__ == "__main__":
    print("I am CHIEF NUMBERS FREEK, The GREAT INDIAN MATH GOD.")
    play = input("\nAre you ready to take the test you called me out for(Yes or No)? ")
    # 如果用户输入的是 "yes"，则调用 game() 函数
    if play.lower() == "yes":
        game()
    # 如果用户输入的不是 "yes"，则打印消息并等待用户输入
    else:
        print("Ok, Nevermind. Let me go back to my great slumber, Bye")
        input()
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```