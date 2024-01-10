# `basic-computer-games\76_Russian_Roulette\python\russianroulette.py`

```
# 导入 random 模块
from random import random

# 定义常量，表示游戏进行的轮数
NUMBER_OF_ROUNDS = 9

# 打印游戏的初始信息
def initial_message() -> None:
    print(" " * 28 + "Russian Roulette")
    print(" " * 15 + "Creative Computing  Morristown, New Jersey\n\n\n")
    print("This is a game of >>>>>>>>>>Russian Roulette.\n")
    print("Here is a Revolver.")

# 解析用户输入的函数
def parse_input() -> int:
    while True:
        try:
            i = int(input("? "))  # 将用户输入转换为整数
            return i
        except ValueError:
            print("Number expected...")  # 捕获输入不是数字的异常并提示用户重新输入

# 游戏的主函数
def main() -> None:
    initial_message()  # 打印初始信息
    while True:
        dead = False  # 标记玩家是否死亡
        n = 0  # 记录玩家进行的轮数
        print("Type '1' to Spin chamber and pull trigger")
        print("Type '2' to Give up")
        print("Go")
        while not dead:
            i = parse_input()  # 解析用户输入

            if i == 2:  # 如果用户输入2，退出游戏
                break

            if random() > 0.8333333333333334:  # 生成随机数，如果大于指定值，玩家死亡
                dead = True
            else:
                print("- CLICK -\n")  # 如果未死亡，打印触发扳机的声音
                n += 1  # 轮数加一

            if n > NUMBER_OF_ROUNDS:  # 如果轮数超过设定值，退出游戏
                break
        if dead:
            print("BANG!!!!!   You're Dead!")  # 如果死亡，打印死亡信息
            print("Condolences will be sent to your relatives.\n\n\n")
            print("...Next victim...")  # 打印下一个受害者信息
        else:
            if n > NUMBER_OF_ROUNDS:  # 如果未死亡且轮数超过设定值，玩家获胜
                print("You win!!!!!")
                print("Let someone else blow his brain out.\n")
            else:
                print("     Chicken!!!!!\n\n\n")  # 如果未死亡且轮数未超过设定值，打印玩家胆小的信息
                print("...Next victim....")  # 打印下一个受害者信息

# 执行主函数
if __name__ == "__main__":
    main()
# 虽然描述说接受 "1" 或 "2"，但原始游戏接受任何数字作为输入，
# 如果不是 "2"，程序会将其视为用户输入了 "1"。这个特性在这个移植版本中保留了。
# 此外，在原始游戏中，你必须"扳动扳机"11次而不是10次才能赢得游戏，
# 因为在开始时 N=0，赢得游戏的条件是 "IF N > 10 THEN  80"。这个问题在这个移植版本中得到了修复，
# 要求用户只扳动扳机十次，尽管回合数可以通过更改常量 NUMBER_OF_ROUNDS 来设置。
#
########################################################
```