# `05_Bagels\python\bagels.py`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 使用 open 函数读取文件内容，'rb' 表示以二进制模式读取文件，BytesIO 用于封装二进制数据成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用 zipfile.ZipFile 创建 ZIP 对象，'r' 表示以只读模式打开 ZIP 文件
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 使用 zip.namelist() 获取 ZIP 文件中的文件名列表，zip.read(n) 读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
import random  # 导入 random 模块，用于生成随机数
from typing import List  # 从 typing 模块中导入 List 类型，用于声明列表类型变量

MAX_GUESSES = 20  # 声明一个常量 MAX_GUESSES，表示最大猜测次数

def print_rules() -> None:  # 定义一个函数 print_rules，返回类型为 None
    print("\nI am thinking of a three-digit number.  Try to guess")  # 打印提示信息
    # 打印猜数字游戏的提示信息
    print("my number and I will give you clues as follows:")
    print("   PICO   - One digit correct but in the wrong position")
    print("   FERMI  - One digit correct and in the right position")
    print("   BAGELS - No digits correct")


def pick_number() -> List[str]:
    # 注意，这里返回的是单独的数字字符串列表，而不是一个整数或字符串
    # 生成一个包含0到9的数字列表
    numbers = list(range(10))
    # 打乱数字列表的顺序
    random.shuffle(numbers)
    # 从打乱后的列表中取出前三个数字
    num = numbers[0:3]
    # 将取出的数字转换成字符串列表
    num_str = [str(i) for i in num]
    return num_str


def get_valid_guess(guesses: int) -> str:
    valid = False
    # 循环直到输入的猜测符合要求
    while not valid:
        # 获取用户输入的猜测
        guess = input(f"Guess # {guesses}     ? ")
        guess = guess.strip()  # 去除字符串两端的空格
        # 猜测必须是三个字符
        if len(guess) == 3:
            # 并且它们应该是数字
            if guess.isnumeric():
                # 并且这些数字必须是唯一的
                if len(set(guess)) == 3:
                    valid = True  # 设置有效标志为True
                else:
                    print("Oh, I forgot to tell you that the number I have in mind")
                    print("has no two digits the same.")  # 打印错误消息
            else:
                print("What?")  # 打印错误消息
        else:
            print("Try guessing a three-digit number.")  # 打印错误消息

    return guess  # 返回猜测的结果


def build_result_string(num: List[str], guess: str) -> str:
    result = ""  # 初始化结果字符串为空

    # Correct digits in wrong place
    # 遍历数字和猜测的数字，如果数字在错误的位置上，则添加"PICO "到结果字符串中
    for i in range(2):
        if num[i] == guess[i + 1]:
            result += "PICO "
        if num[i + 1] == guess[i]:
            result += "PICO "
    if num[0] == guess[2]:
        result += "PICO "
    if num[2] == guess[0]:
        result += "PICO "

    # Correct digits in right place
    # 遍历数字和猜测的数字，如果数字在正确的位置上，则添加"FERMI "到结果字符串中
    for i in range(3):
        if num[i] == guess[i]:
            result += "FERMI "

    # Nothing right?
    # 如果结果字符串为空，则表示没有猜对任何数字
    if result == "":
        result = "BAGELS"
    # 将变量result赋值为字符串"BAGELS"

    return result
    # 返回result变量的值作为函数的结果


def main() -> None:
    # Intro text
    print("\n                Bagels")
    # 打印游戏标题
    print("Creative Computing  Morristown, New Jersey\n\n")
    # 打印游戏的创作信息

    # Anything other than N* will show the rules
    response = input("Would you like the rules (Yes or No)? ")
    # 获取用户输入的是否需要查看游戏规则
    if len(response) > 0:
        if response.upper()[0] != "N":
            # 如果用户输入的第一个字符不是N，则打印游戏规则
            print_rules()
    else:
        # 如果用户没有输入或者输入为空，则打印游戏规则
        print_rules()

    games_won = 0
    # 初始化游戏胜利次数为0
    still_running = True
    # 初始化游戏运行状态为True
    while still_running:  # 当 still_running 为真时执行循环

        # New round
        num = pick_number()  # 调用 pick_number() 函数生成一个新的数字
        num_str = "".join(num)  # 将数字转换为字符串
        guesses = 1  # 初始化猜测次数为1

        print("\nO.K.  I have a number in mind.")  # 打印消息
        guessing = True  # 设置 guessing 变量为真
        while guessing:  # 当 guessing 为真时执行循环

            guess = get_valid_guess(guesses)  # 调用 get_valid_guess() 函数获取有效的猜测

            if guess == num_str:  # 如果猜测正确
                print("You got it!!!\n")  # 打印消息
                games_won += 1  # 游戏胜利次数加一
                guessing = False  # 设置 guessing 变量为假，结束循环
            else:  # 如果猜测错误
                print(build_result_string(num, guess))  # 打印猜测结果
                guesses += 1  # 猜测次数加一
                if guesses > MAX_GUESSES:  # 如果猜测次数超过最大次数
                    print("Oh well")  # 打印"哦，好吧"
                    print(f"That's {MAX_GUESSES} guesses.  My number was {num_str}")  # 打印已经达到最大猜测次数和正确数字
                    guessing = False  # 将猜测状态设置为False，结束猜数字游戏

        valid_response = False  # 初始化有效回答为False
        while not valid_response:  # 当回答无效时
            response = input("Play again (Yes or No)? ")  # 获取用户输入的是否再玩一次的回答
            if len(response) > 0:  # 如果回答不为空
                valid_response = True  # 将有效回答设置为True
                if response.upper()[0] != "Y":  # 如果回答的第一个字母不是Y
                    still_running = False  # 将游戏状态设置为False，结束游戏

    if games_won > 0:  # 如果赢得了游戏
        print(f"\nA {games_won} point Bagels buff!!")  # 打印赢得游戏的得分

    print("Hope you had fun.  Bye.\n")  # 打印祝愿玩家玩得开心，再见


if __name__ == "__main__":  # 如果当前文件被直接运行
    main()
```
调用名为main的函数。

```python
######################################################################
#
# Porting Notes
#
#   The original program did an unusually good job of validating the
#   player's input (compared to many of the other programs in the
#   book). Those checks and responses have been exactly reproduced.
#
#
# Ideas for Modifications
#
#   It should probably mention that there's a maximum of MAX_NUM
#   guesses in the instructions somewhere, shouldn't it?
#
#   Could this program be written to use anywhere from, say 2 to 6
#   digits in the number to guess? How would this change the routine
#   that creates the "result" string?
#
```
这部分是注释，提供了一些关于程序的说明和修改的想法。
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制数据，并封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流创建一个ZIP文件对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历ZIP文件中的文件名列表，读取每个文件的数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭ZIP文件对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```