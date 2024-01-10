# `basic-computer-games\29_Craps\python\craps.py`

```
#!/usr/bin/env python3
"""This game simulates the games of craps played according to standard Nevada craps table rules.

That is:

1. A 7 or 11 on the first roll wins
2. A 2, 3, or 12 on the first roll loses
3. Any other number rolled becomes your "point." You continue to roll; if you get your point you win. If you
   roll a 7, you lose and the dice change hands when this happens.

This version of craps was modified by Steve North of Creative Computing. It is based on an original which
appeared one day one a computer at DEC.
"""
from random import randint


def throw_dice() -> int:
    # 通过随机生成两个骰子的点数之和来模拟掷骰子的结果
    return randint(1, 6) + randint(1, 6)


def main() -> None:
    # 打印游戏标题
    print(" " * 33 + "Craps")
    # 打印游戏信息
    print(" " * 15 + "Creative Computing  Morristown, New Jersey\n\n\n")

    # 初始化玩家的赢钱数
    winnings = 0
    # 打印游戏规则
    print("2,3,12 are losers; 4,5,6,8,9,10 are points; 7,11 are natural winners.")

    # 设置是否继续游戏的标志
    play_again = True
    # 当 play_again 为 True 时，进入循环
    while play_again:
        # 从用户输入中获取赌注金额
        wager = int(input("Input the amount of your wager: "))
    
        # 打印信息，提示将要投掷骰子
        print("I will now throw the dice")
        # 投掷骰子，获取第一次的点数
        roll_1 = throw_dice()
    
        # 如果第一次点数为 7 或 11，则赢得赌注金额
        if roll_1 in [7, 11]:
            print(f"{roll_1} - natural.... a winner!!!!")
            print(f"{roll_1} pays even money, you win {wager} dollars")
            winnings += wager
        # 如果第一次点数为 2，则输掉赌注金额
        elif roll_1 == 2:
            print(f"{roll_1} - snake eyes.... you lose.")
            print(f"You lose {wager} dollars")
            winnings -= wager
        # 如果第一次点数为 3 或 12，则输掉赌注金额
        elif roll_1 in [3, 12]:
            print(f"{roll_1} - craps.... you lose.")
            print(f"You lose {wager} dollars")
            winnings -= wager
        # 如果第一次点数为其它点数，则继续投掷骰子
        else:
            print(f"{roll_1} is the point. I will roll again")
            roll_2 = 0
            # 当第二次点数不是第一次点数或者 7 时，继续投掷骰子
            while roll_2 not in [roll_1, 7]:
                roll_2 = throw_dice()
                # 如果第二次点数为 7，则输掉赌注金额
                if roll_2 == 7:
                    print(f"{roll_2} - craps. You lose.")
                    print(f"You lose $ {wager}")
                    winnings -= wager
                # 如果第二次点数等于第一次点数，则赢得赌注金额的两倍
                elif roll_2 == roll_1:
                    print(f"{roll_1} - a winner.........congrats!!!!!!!!")
                    print(
                        f"{roll_1} at 2 to 1 odds pays you...let me see... {2 * wager} dollars"
                    )
                    winnings += 2 * wager
                else:
                    print(f"{roll_2} - no point. I will roll again")
    
        # 询问用户是否想再玩一次
        m = input("  If you want to play again print 5 if not print 2: ")
        # 根据赢得的金额情况，打印相应的信息
        if winnings < 0:
            print(f"You are now under ${-winnings}")
        elif winnings > 0:
            print(f"You are now ahead ${winnings}")
        else:
            print("You are now even at 0")
        # 根据用户输入判断是否继续玩游戏
        play_again = m == "5"
    
    # 根据赢得的金额情况，打印最终的结果信息
    if winnings < 0:
        print("Too bad, you are in the hole. Come again.")
    elif winnings > 0:
        print("Congratulations---you came out a winner. Come again."
    # 如果以上条件都不满足，则执行以下代码
    else:
        # 打印恭喜消息，表示玩家达到了平局
        print("Congratulations---you came out even, not bad for an amateur")
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```