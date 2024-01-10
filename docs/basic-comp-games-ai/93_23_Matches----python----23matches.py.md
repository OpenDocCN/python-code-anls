# `basic-computer-games\93_23_Matches\python\23matches.py`

```
#!/usr/bin/env python3
# 指定脚本解释器为 Python 3
# 23 Matches
# 游戏名称

# Converted from BASIC to Python by Trevor Hobson
# 由 Trevor Hobson 将 BASIC 语言转换为 Python

import random
# 导入 random 模块

def play_game() -> None:
    """Play one round of the game"""
    # 定义函数，用于进行一轮游戏

    matches = 23
    # 初始化游戏中的火柴数量为 23
    humans_turn = random.randint(0, 1) == 1
    # 随机决定玩家是否先手
    if humans_turn:
        # 如果玩家先手
        print("Tails! You go first.\n")
        # 输出提示信息
        prompt_human = "How many do you wish to remove "
        # 设置玩家输入移除火柴数量的提示信息
    else:
        # 如果玩家不先手
        print("Heads! I win! Ha! Ha!")
        # 输出提示信息
        print("Prepare to lose, meatball-nose!!")
        # 输出提示信息

    choice_human = 2
    # 初始化玩家的选择为 2
    # 当还有火柴时进行循环
    while matches > 0:
        # 如果是人类的回合
        if humans_turn:
            # 人类选择火柴数量的初始化
            choice_human = 0
            # 如果只剩下一根火柴，人类只能拿走一根
            if matches == 1:
                choice_human = 1
            # 当人类选择为0时，循环直到输入合法的选择
            while choice_human == 0:
                try:
                    # 人类输入拿走的火柴数量
                    choice_human = int(input(prompt_human))
                    # 如果选择不在[1, 2, 3]之间或者大于剩余火柴数量，重新输入
                    if choice_human not in [1, 2, 3] or choice_human > matches:
                        choice_human = 0
                        print("Very funny! Dummy!")
                        print("Do you want to play or goof around?")
                        prompt_human = "Now, how many matches do you want "
                except ValueError:
                    # 如果输入不是数字，提示重新输入
                    print("Please enter a number.")
                    prompt_human = "How many do you wish to remove "
            # 更新剩余火柴数量
            matches = matches - choice_human
            # 如果没有剩余火柴，人类输了
            if matches == 0:
                print("You poor boob! You took the last match! I gotcha!!")
                print("Ha ! Ha ! I beat you !!\n")
                print("Good bye loser!")
            else:
                # 打印剩余火柴数量
                print("There are now", matches, "matches remaining.\n")
        # 如果是电脑的回合
        else:
            # 电脑根据人类选择拿走火柴
            choice_computer = 4 - choice_human
            # 如果只剩下一根火柴，电脑只能拿走一根
            if matches == 1:
                choice_computer = 1
            # 如果剩余火柴在1和3之间，电脑拿走剩余火柴数量-1根
            elif 1 < matches < 4:
                choice_computer = matches - 1
            # 更新剩余火柴数量
            matches = matches - choice_computer
            # 如果没有剩余火柴，人类赢了
            if matches == 0:
                print("You won, floppy ears !")
                print("Think you're pretty smart !")
                print("Let's play again and I'll blow your shoes off !!")
            else:
                # 打印电脑拿走的火柴数量和剩余火柴数量
                print("My turn ! I remove", choice_computer, "matches")
                print("The number of matches is now", matches, "\n")
        # 切换回合
        humans_turn = not humans_turn
        prompt_human = "Your turn -- you may take 1, 2 or 3 matches.\nHow many do you wish to remove "
# 定义主函数，不返回任何结果
def main() -> None:
    # 打印标题
    print(" " * 31 + "23 MATCHHES")
    # 打印创意计算公司信息
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")
    # 打印游戏介绍
    print("This is a game called '23 Matches'.\n")
    print("When it is your turn, you may take one, two, or three")
    print("matches. The object of the game is not to have to take")
    print("the last match.\n")
    print("Let's flip a coin to see who goes first.")
    print("If it comes up heads, I will win the toss.\n")

    # 初始化变量，用于控制游戏是否继续进行
    keep_playing = True
    # 循环进行游戏
    while keep_playing:
        # 调用游戏函数
        play_game()
        # 询问是否继续游戏
        keep_playing = input("\nPlay again? (yes or no) ").lower().startswith("y")

# 如果当前脚本为主程序，则执行主函数
if __name__ == "__main__":
    main()
```