# `d:/src/tocomm/basic-computer-games\93_23_Matches\python\23matches.py`

```
#!/usr/bin/env python3  # 指定脚本解释器为 Python3
# 23 Matches  # 程序名称或简要描述
#
# Converted from BASIC to Python by Trevor Hobson  # 作者信息

import random  # 导入 random 模块


def play_game() -> None:  # 定义函数 play_game，返回类型为 None
    """Play one round of the game"""  # 函数的文档字符串

    matches = 23  # 初始化变量 matches 为 23
    humans_turn = random.randint(0, 1) == 1  # 随机生成 0 或 1，判断是否轮到玩家
    if humans_turn:  # 如果轮到玩家
        print("Tails! You go first.\n")  # 打印提示信息
        prompt_human = "How many do you wish to remove "  # 初始化提示信息
    else:  # 如果轮到电脑
        print("Heads! I win! Ha! Ha!")  # 打印提示信息
        print("Prepare to lose, meatball-nose!!")  # 打印提示信息
    choice_human = 2  # 初始化玩家选择的变量为2
    while matches > 0:  # 当还有火柴时进行循环
        if humans_turn:  # 如果轮到玩家
            choice_human = 0  # 初始化玩家选择为0
            if matches == 1:  # 如果只剩下一根火柴
                choice_human = 1  # 玩家必须选择拿走最后一根火柴
            while choice_human == 0:  # 当玩家选择为0时进行循环
                try:  # 尝试获取玩家输入
                    choice_human = int(input(prompt_human))  # 获取玩家输入的火柴数量
                    if choice_human not in [1, 2, 3] or choice_human > matches:  # 如果玩家输入不合法
                        choice_human = 0  # 重置玩家选择为0
                        print("Very funny! Dummy!")  # 输出错误提示
                        print("Do you want to play or goof around?")  # 提示玩家重新选择
                        prompt_human = "Now, how many matches do you want "  # 更新提示信息
                except ValueError:  # 如果玩家输入不是数字
                    print("Please enter a number.")  # 提示玩家输入数字
                    prompt_human = "How many do you wish to remove "  # 更新提示信息
            matches = matches - choice_human  # 更新剩余火柴数量
            if matches == 0:  # 如果没有剩余火柴
                print("You poor boob! You took the last match! I gotcha!!")  # 输出玩家失败的信息
                print("Ha ! Ha ! I beat you !!\n")  # 打印消息，表示计算机赢了
                print("Good bye loser!")  # 打印消息，表示计算机赢了
            else:
                print("There are now", matches, "matches remaining.\n")  # 打印消息，显示剩余的火柴数量
        else:
            choice_computer = 4 - choice_human  # 计算计算机的选择，使得剩余的火柴数量为4
            if matches == 1:
                choice_computer = 1  # 如果只剩下一根火柴，计算机选择拿走一根
            elif 1 < matches < 4:
                choice_computer = matches - 1  # 如果剩余火柴数量在1和3之间，计算机选择拿走剩余数量减1根
            matches = matches - choice_computer  # 更新剩余的火柴数量
            if matches == 0:
                print("You won, floppy ears !")  # 打印消息，表示玩家赢了
                print("Think you're pretty smart !")  # 打印消息，表示玩家赢了
                print("Let's play again and I'll blow your shoes off !!")  # 打印消息，表示玩家赢了
            else:
                print("My turn ! I remove", choice_computer, "matches")  # 打印消息，显示计算机拿走的火柴数量
                print("The number of matches is now", matches, "\n")  # 打印消息，显示剩余的火柴数量
        humans_turn = not humans_turn  # 切换玩家回合
        prompt_human = "Your turn -- you may take 1, 2 or 3 matches.\nHow many do you wish to remove "  # 提示玩家可以拿走1、2或3根火柴
def main() -> None:
    # 打印游戏标题
    print(" " * 31 + "23 MATCHHES")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    print("This is a game called '23 Matches'.\n")
    print("When it is your turn, you may take one, two, or three")
    print("matches. The object of the game is not to have to take")
    print("the last match.\n")
    print("Let's flip a coin to see who goes first.")
    print("If it comes up heads, I will win the toss.\n")

    keep_playing = True
    while keep_playing:
        # 调用 play_game() 函数来进行游戏
        play_game()
        # 询问玩家是否继续游戏
        keep_playing = input("\nPlay again? (yes or no) ").lower().startswith("y")


if __name__ == "__main__":
    # 调用 main() 函数来开始游戏
    main()
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