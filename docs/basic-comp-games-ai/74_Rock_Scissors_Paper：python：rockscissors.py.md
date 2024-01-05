# `74_Rock_Scissors_Paper\python\rockscissors.py`

```
#!/usr/bin/env python3
# ROCKSCISSORS
#
# Converted from BASIC to Python by Trevor Hobson

import random


def play_game() -> None:
    """Play one round of the game"""

    while True:
        try:
            # 提示用户输入游戏次数，并将输入转换为整数
            games = int(input("How many games? "))
            # 如果输入的游戏次数大于等于11，则提示不允许玩这么多次
            if games >= 11:
                print("Sorry, but we aren't allowed to play that many.")
            else:
                break

        except ValueError:
    # 初始化计数器，记录电脑和人的胜利次数
    won_computer = 0
    won_human = 0

    # 循环进行游戏
    for game in range(games):
        # 打印游戏编号
        print("\nGame number", game + 1)
        # 电脑随机生成猜拳结果
        guess_computer = random.randint(1, 3)
        # 打印提示信息，告诉玩家石头、剪刀、布对应的数字
        print("3=Rock...2=Scissors...1=Paper")
        # 初始化玩家的猜拳结果
        guess_human = 0
        # 玩家输入猜拳结果，直到输入合法的数字
        while guess_human == 0:
            try:
                guess_human = int(input("1...2...3...What's your choice "))
                # 如果输入的数字不在1、2、3中，重新输入
                if guess_human not in [1, 2, 3]:
                    guess_human = 0
                    print("Invalid")

            # 如果输入的不是数字，提示玩家重新输入
            except ValueError:
                print("Please enter a number.")
        # 打印电脑的选择
        print("This is my choice...")
        if guess_computer == 1:  # 如果计算机猜测为1，表示出Paper
            print("...Paper")
        elif guess_computer == 2:  # 如果计算机猜测为2，表示出Scissors
            print("...Scissors")
        elif guess_computer == 3:  # 如果计算机猜测为3，表示出Rock
            print("...Rock")
        if guess_computer == guess_human:  # 如果计算机和玩家猜测相同，平局
            print("Tie Game. No winner")
        elif guess_computer > guess_human:  # 如果计算机猜测大于玩家
            if guess_human != 1 or guess_computer != 3:  # 且玩家不是Paper或者计算机不是Rock
                print("Wow! I win!!!")  # 计算机获胜
                won_computer = won_computer + 1  # 计算机获胜次数加一
            else:
                print("You win!!!")  # 玩家获胜
                won_human = won_human + 1  # 玩家获胜次数加一
        elif guess_computer == 1:  # 如果计算机猜测为1，表示出Paper
            if guess_human != 3:  # 且玩家不是Rock
                print("You win!!!")  # 玩家获胜
                won_human = won_human + 1  # 玩家获胜次数加一
            else:
                print("Wow! I win!!!")  # 打印出计算机赢得游戏的消息
                won_computer = won_computer + 1  # 增加计算机赢得游戏的计数

    print("\nHere is the final game score:")  # 打印出最终游戏得分的消息
    print("I have won", won_computer, "game(s).")  # 打印出计算机赢得的游戏数
    print("You have won", won_human, "game(s).")  # 打印出玩家赢得的游戏数
    print("and", games - (won_computer + won_human), "game(s) ended in a tie.")  # 打印出平局的游戏数
    print("\nThanks for playing!!\n")  # 打印出感谢玩家参与游戏的消息


def main() -> None:
    print(" " * 21 + "GAME OF ROCK, SCISSORS, PAPER")  # 打印出游戏标题
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")  # 打印出游戏信息

    keep_playing = True  # 初始化变量，用于控制游戏是否继续进行
    while keep_playing:

        play_game()  # 调用游戏进行函数

        keep_playing = input("Play again? (yes or no) ").lower().startswith("y")  # 根据玩家输入判断是否继续游戏
# 如果当前脚本被直接执行而不是被导入，则执行main函数
if __name__ == "__main__":
    main()
```