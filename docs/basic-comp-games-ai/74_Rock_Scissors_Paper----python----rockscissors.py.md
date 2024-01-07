# `basic-computer-games\74_Rock_Scissors_Paper\python\rockscissors.py`

```

#!/usr/bin/env python3
# ROCKSCISSORS
#
# Converted from BASIC to Python by Trevor Hobson

import random


def play_game() -> None:
    """Play one round of the game"""

    # 循环直到输入有效的游戏次数
    while True:
        try:
            games = int(input("How many games? "))
            if games >= 11:
                print("Sorry, but we aren't allowed to play that many.")
            else:
                break

        except ValueError:
            print("Please enter a number.")

    won_computer = 0
    won_human = 0

    # 开始游戏循环
    for game in range(games):
        print("\nGame number", game + 1)
        # 计算计算机的选择
        guess_computer = random.randint(1, 3)
        print("3=Rock...2=Scissors...1=Paper")
        guess_human = 0
        # 循环直到输入有效的玩家选择
        while guess_human == 0:
            try:
                guess_human = int(input("1...2...3...What's your choice "))
                if guess_human not in [1, 2, 3]:
                    guess_human = 0
                    print("Invalid")

            except ValueError:
                print("Please enter a number.")
        print("This is my choice...")
        # 打印计算机的选择
        if guess_computer == 1:
            print("...Paper")
        elif guess_computer == 2:
            print("...Scissors")
        elif guess_computer == 3:
            print("...Rock")
        # 判断游戏结果
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
    # 打印游戏结果
    print("\nHere is the final game score:")
    print("I have won", won_computer, "game(s).")
    print("You have won", won_human, "game(s).")
    print("and", games - (won_computer + won_human), "game(s) ended in a tie.")
    print("\nThanks for playing!!\n")


def main() -> None:
    print(" " * 21 + "GAME OF ROCK, SCISSORS, PAPER")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")

    keep_playing = True
    # 循环直到玩家选择不再继续游戏
    while keep_playing:

        play_game()

        keep_playing = input("Play again? (yes or no) ").lower().startswith("y")


if __name__ == "__main__":
    main()

```