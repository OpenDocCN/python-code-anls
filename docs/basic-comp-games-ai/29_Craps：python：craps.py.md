# `d:/src/tocomm/basic-computer-games\29_Craps\python\craps.py`

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
from random import randint  # 导入 randint 函数


def throw_dice() -> int:  # 定义名为 throw_dice 的函数，返回值为整数类型
    return randint(1, 6) + randint(1, 6)  # 返回两次掷骰子的结果之和
def main() -> None:
    # 打印游戏标题
    print(" " * 33 + "Craps")
    # 打印游戏信息
    print(" " * 15 + "Creative Computing  Morristown, New Jersey\n\n\n")

    # 初始化赢得的金额
    winnings = 0
    # 打印游戏规则
    print("2,3,12 are losers; 4,5,6,8,9,10 are points; 7,11 are natural winners.")

    # 初始化是否继续游戏的标志
    play_again = True
    # 循环进行游戏
    while play_again:
        # 获取玩家下注金额
        wager = int(input("Input the amount of your wager: "))

        # 打印掷骰子的信息
        print("I will now throw the dice")
        # 第一次掷骰子
        roll_1 = throw_dice()

        # 判断第一次掷骰子的结果
        if roll_1 in [7, 11]:
            # 如果是自然胜，打印胜利信息并计算赢得的金额
            print(f"{roll_1} - natural.... a winner!!!!")
            print(f"{roll_1} pays even money, you win {wager} dollars")
            winnings += wager
        elif roll_1 == 2:
            # 如果是2点，打印失败信息
            print(f"{roll_1} - snake eyes.... you lose.")
            print(f"You lose {wager} dollars")  # 打印玩家输掉的赌注金额
            winnings -= wager  # 从总赢得金额中减去玩家输掉的赌注金额
        elif roll_1 in [3, 12]:  # 如果第一次掷骰子的点数为3或12
            print(f"{roll_1} - craps.... you lose.")  # 打印掷骰子结果为3或12时玩家输掉的消息
            print(f"You lose {wager} dollars")  # 打印玩家输掉的赌注金额
            winnings -= wager  # 从总赢得金额中减去玩家输掉的赌注金额
        else:  # 如果第一次掷骰子的点数不是2、3、12
            print(f"{roll_1} is the point. I will roll again")  # 打印第一次掷骰子的点数，并提示将再次掷骰子
            roll_2 = 0  # 初始化第二次掷骰子的点数
            while roll_2 not in [roll_1, 7]:  # 当第二次掷骰子的点数不等于第一次点数或为7时
                roll_2 = throw_dice()  # 执行掷骰子的函数，得到第二次掷骰子的点数
                if roll_2 == 7:  # 如果第二次掷骰子的点数为7
                    print(f"{roll_2} - craps. You lose.")  # 打印掷骰子结果为7时玩家输掉的消息
                    print(f"You lose $ {wager}")  # 打印玩家输掉的赌注金额
                    winnings -= wager  # 从总赢得金额中减去玩家输掉的赌注金额
                elif roll_2 == roll_1:  # 如果第二次掷骰子的点数等于第一次点数
                    print(f"{roll_1} - a winner.........congrats!!!!!!!!")  # 打印掷骰子结果为第一次点数时玩家赢得的消息
                    print(
                        f"{roll_1} at 2 to 1 odds pays you...let me see... {2 * wager} dollars"
                    )  # 打印根据赔率计算出的玩家赢得的金额
# 如果赢得了游戏，将赌注的两倍加到总赢得金额中
winnings += 2 * wager
# 如果没有点数，打印信息并重新掷骰子
print(f"{roll_2} - no point. I will roll again")

# 要求用户输入是否想再玩一次
m = input("  If you want to play again print 5 if not print 2: ")
# 如果总赢得金额为负数，打印欠款信息
print(f"You are now under ${-winnings}")
# 如果总赢得金额为正数，打印盈利信息
print(f"You are now ahead ${winnings}")
# 如果总赢得金额为零，打印平局信息
print("You are now even at 0")
# 判断用户是否想再玩一次
play_again = m == "5"

# 如果总赢得金额为负数，打印输家信息
print("Too bad, you are in the hole. Come again.")
# 如果总赢得金额为正数，打印赢家信息
print("Congratulations---you came out a winner. Come again.")
# 如果总赢得金额为零，打印平局信息
print("Congratulations---you came out even, not bad for an amateur")
# 如果当前脚本被直接执行，则调用 main() 函数。
```