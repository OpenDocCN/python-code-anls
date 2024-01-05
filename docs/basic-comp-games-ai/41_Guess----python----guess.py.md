# `41_Guess\python\guess.py`

```
"""
Guess

From: Basic Computer Games (1978)

 "In program Guess, the computer  chooses a random
  integer between 0 and any limit and any limit you
  set. You must then try to guess the number the
  computer has choosen using the clues provideed by
  the computer.
   You should be able to guess the number in one less
  than the number of digits needed to  represent the
  number in binary notation - i.e. in base 2. This ought
  to give you a clue as to the optimum search technique.
   Guess converted from the original program in FOCAL
  which appeared in the book "Computers in the Classroom"
  by Walt Koetke of Lexington High School, Lexington,
  Massaschusetts.
"""
# 上面是一个多行注释，用三个双引号包裹起来，用于解释程序的功能和来源。
# 导入 log 和 random 模块
from math import log
from random import random
# 导入 Tuple 类型
from typing import Tuple

# 定义一个没有返回值的函数，用于在控制台输出多个空行
def insert_whitespaces() -> None:
    print("\n\n\n\n\n")

# 定义一个返回元组的函数，用于设置猜测数字的范围
def limit_set() -> Tuple[int, int]:
    print("                   Guess")
    print("Creative Computing  Morristown, New Jersey")
    print("\n\n\n")
    print("This is a number guessing game. I'll think")
    print("of a number between 1 and any limit you want.\n")  # 打印提示信息
    print("Then you have to guess what it is\n")  # 打印提示信息
    print("What limit do you want?")  # 打印提示信息

    limit = int(input())  # 从用户输入中获取限制值并转换为整数

    while limit <= 0:  # 当限制值小于等于0时
        print("Please insert a number greater or equal to 1")  # 打印提示信息
        limit = int(input())  # 重新从用户输入中获取限制值并转换为整数

    # limit_goal = Number of digits "limit" in binary has
    limit_goal = int((log(limit) / log(2)) + 1)  # 计算限制值在二进制中的位数

    return limit, limit_goal  # 返回限制值和计算得到的位数


def main() -> None:  # 主函数声明
    limit, limit_goal = limit_set()  # 调用limit_set函数获取限制值和计算得到的位数
    while True:  # 无限循环
        guess_count = 1  # 初始化猜测次数为1
        still_guessing = True  # 初始化变量still_guessing为True，表示猜测游戏仍在进行中
        won = False  # 初始化变量won为False，表示游戏尚未获胜
        my_guess = int(limit * random() + 1)  # 生成一个1到limit之间的随机整数作为程序员所猜的数字

        print(f"I'm thinking of a number between 1 and {limit}")  # 打印程序员所思考的数字范围
        print("Now you try to guess what it is.")  # 提示玩家开始猜数字

        while still_guessing:  # 进入循环，直到猜测游戏结束
            n = int(input())  # 从用户输入中获取一个整数作为猜测的数字

            if n < 0:  # 如果用户输入的数字小于0
                break  # 退出循环

            insert_whitespaces()  # 调用insert_whitespaces函数，可能是用于插入空白字符
            if n < my_guess:  # 如果用户猜测的数字小于程序员所猜的数字
                print("Too low. Try a bigger answer")  # 提示用户猜测的数字太小
                guess_count += 1  # 猜测次数加1
            elif n > my_guess:  # 如果用户猜测的数字大于程序员所猜的数字
                print("Too high. Try a smaller answer")  # 提示用户猜测的数字太大
                guess_count += 1  # 猜测次数加1
            else:
                # 打印猜对的次数，并结束猜词游戏
                print(f"That's it! You got it in {guess_count} tries")
                won = True
                still_guessing = False

        if won:
            # 如果猜对了
            if guess_count < limit_goal:
                # 如果猜对的次数小于目标次数，打印"Very good."
                print("Very good.")
            elif guess_count == limit_goal:
                # 如果猜对的次数等于目标次数，打印"Good."
                print("Good.")
            else:
                # 如果猜对的次数大于目标次数，打印应该只需要多少次猜对
                print(f"You should have been able to get it in only {limit_goal}")
            # 插入空行
            insert_whitespaces()
        else:
            # 如果没有猜对，插入空行并设置新的猜词游戏限制
            insert_whitespaces()
            limit, limit_goal = limit_set()


if __name__ == "__main__":
    # 调用主函数
    main()
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```