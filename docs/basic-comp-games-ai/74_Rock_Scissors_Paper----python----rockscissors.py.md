# `basic-computer-games\74_Rock_Scissors_Paper\python\rockscissors.py`

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
            # 询问玩几局游戏
            games = int(input("How many games? "))
            # 如果游戏局数大于等于11，则提示不允许玩那么多局
            if games >= 11:
                print("Sorry, but we aren't allowed to play that many.")
            else:
                break

        except ValueError:
            print("Please enter a number.")

    won_computer = 0
    won_human = 0

    for game in range(games):
        print("\nGame number", game + 1)
        # 计算电脑的猜拳结果
        guess_computer = random.randint(1, 3)
        print("3=Rock...2=Scissors...1=Paper")
        guess_human = 0
        while guess_human == 0:
            try:
                # 玩家输入猜拳结果
                guess_human = int(input("1...2...3...What's your choice "))
                # 如果玩家输入不在1、2、3之间，则重新输入
                if guess_human not in [1, 2, 3]:
                    guess_human = 0
                    print("Invalid")

            except ValueError:
                print("Please enter a number.")
        print("This is my choice...")
        # 根据电脑的猜拳结果输出对应的选择
        if guess_computer == 1:
            print("...Paper")
        elif guess_computer == 2:
            print("...Scissors")
        elif guess_computer == 3:
            print("...Rock")
        # 判断输赢
        if guess_computer == guess_human:
            print("Tie Game. No winner")
        elif guess_computer > guess_human:
            if guess_human != 1 or guess_computer != 3:
                print("Wow! I win!!!")
                won_computer = won_computer + 1
            else:
                print("You win!!!")
                won_human = won_human + 1
        elif guess_computer == 1:
            if guess_human != 3:
                print("You win!!!")
                won_human = won_human + 1
            else:
                print("Wow! I win!!!")
                won_computer = won_computer + 1
    print("\nHere is the final game score:")
    print("I have won", won_computer, "game(s).")
    # 打印玩家赢得的游戏数
    print("You have won", won_human, "game(s).")
    # 打印平局游戏数
    print("and", games - (won_computer + won_human), "game(s) ended in a tie.")
    # 打印结束语
    print("\nThanks for playing!!\n")
# 定义主函数，不返回任何结果
def main() -> None:
    # 打印游戏标题
    print(" " * 21 + "GAME OF ROCK, SCISSORS, PAPER")
    # 打印游戏作者信息
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")

    # 初始化游戏继续标志为 True
    keep_playing = True
    # 当游戏继续标志为 True 时，循环执行游戏
    while keep_playing:

        # 调用游戏函数
        play_game()

        # 获取用户输入，判断是否继续游戏
        keep_playing = input("Play again? (yes or no) ").lower().startswith("y")

# 如果当前脚本为主程序，则执行主函数
if __name__ == "__main__":
    main()
```