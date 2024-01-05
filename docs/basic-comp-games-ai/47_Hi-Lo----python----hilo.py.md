# `47_Hi-Lo\python\hilo.py`

```
#!/usr/bin/env python3  # 指定使用 Python3 解释器来执行脚本

import random  # 导入 random 模块

MAX_ATTEMPTS = 6  # 设置最大尝试次数为 6
QUESTION_PROMPT = "? "  # 设置问题提示符为 "?"

def main() -> None:  # 定义主函数，返回类型为 None
    print("HI LO")  # 打印 "HI LO"
    print("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")  # 打印创意计算机的信息
    print("THIS IS THE GAME OF HI LO.\n")  # 打印游戏介绍
    print("YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE")  # 打印玩家有6次机会猜测奖池中的金额
    print("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU")  # 打印奖池金额在1到100美元之间
    print("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!")  # 打印如果猜对了，就赢得奖池中的所有金额
    print("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,")  # 打印然后有机会赢得更多的钱
    print("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.\n\n")  # 打印如果没有猜对，游戏结束

    total_winnings = 0  # 初始化总赢得金额为 0
    while True:  # 进入无限循环
        print()  # 打印空行
        # 生成一个1到100之间的随机整数作为秘密数字
        secret = random.randint(1, 100)
        # 初始化猜对标志为False
        guessed_correctly = False

        # 循环最多MAX_ATTEMPTS次
        for _attempt in range(MAX_ATTEMPTS):
            # 打印提示信息，等待用户输入猜测的数字
            print("YOUR GUESS", end=QUESTION_PROMPT)
            guess = int(input())

            # 如果猜对了，打印获胜信息，设置猜对标志为True，跳出循环
            if guess == secret:
                print(f"GOT IT!!!!!!!!!!   YOU WIN {secret} DOLLARS.")
                guessed_correctly = True
                break
            # 如果猜测的数字大于秘密数字，打印提示信息
            elif guess > secret:
                print("YOUR GUESS IS TOO HIGH.")
            # 如果猜测的数字小于秘密数字，打印提示信息
            else:
                print("YOUR GUESS IS TOO LOW.")

        # 如果猜对了，增加总奖金数，打印总奖金数
        if guessed_correctly:
            total_winnings += secret
            print(f"YOUR TOTAL WINNINGS ARE NOW {total_winnings} DOLLARS.")
        # 如果没有猜对，执行其他操作
        else:
        print(f"YOU BLEW IT...TOO BAD...THE NUMBER WAS {secret}")  # 打印玩家猜错时的提示信息，显示正确答案

    print("\n")  # 打印一个空行
    print("PLAY AGAIN (YES OR NO)", end=QUESTION_PROMPT)  # 打印提示信息，询问玩家是否要再玩一次
    answer = input().upper()  # 获取玩家输入的答案，并转换为大写
    if answer != "YES":  # 如果玩家的答案不是YES
        break  # 退出循环，结束游戏

print("\nSO LONG.  HOPE YOU ENJOYED YOURSELF!!!")  # 打印结束游戏的提示信息


if __name__ == "__main__":  # 如果当前文件被直接运行
    main()  # 调用main函数开始游戏
```