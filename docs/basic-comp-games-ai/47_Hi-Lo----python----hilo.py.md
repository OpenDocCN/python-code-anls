# `basic-computer-games\47_Hi-Lo\python\hilo.py`

```

#!/usr/bin/env python3
# 指定使用 Python3 解释器

import random
# 导入 random 模块

MAX_ATTEMPTS = 6
# 最大尝试次数
QUESTION_PROMPT = "? "
# 问题提示符

def main() -> None:
    # 主函数

    print("HI LO")
    print("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    print("THIS IS THE GAME OF HI LO.\n")
    print("YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE")
    print("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU")
    print("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!")
    print("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,")
    print("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.\n\n")
    # 打印游戏介绍

    total_winnings = 0
    # 总奖金初始化为 0
    while True:
        # 无限循环

        print()
        # 打印空行
        secret = random.randint(1, 100)
        # 生成 1 到 100 之间的随机数作为秘密数字
        guessed_correctly = False
        # 初始化猜对标志为 False

        for _attempt in range(MAX_ATTEMPTS):
            # 循环最大尝试次数次

            print("YOUR GUESS", end=QUESTION_PROMPT)
            # 打印提示
            guess = int(input())
            # 获取用户输入的猜测

            if guess == secret:
                # 如果猜对了
                print(f"GOT IT!!!!!!!!!!   YOU WIN {secret} DOLLARS.")
                # 打印猜对的消息
                guessed_correctly = True
                # 设置猜对标志为 True
                break
                # 退出循环
            elif guess > secret:
                # 如果猜的数字太大
                print("YOUR GUESS IS TOO HIGH.")
                # 打印猜测数字太大的消息
            else:
                # 如果猜的数字太小
                print("YOUR GUESS IS TOO LOW.")
                # 打印猜测数字太小的消息

        if guessed_correctly:
            # 如果猜对了
            total_winnings += secret
            # 总奖金增加猜对的数字
            print(f"YOUR TOTAL WINNINGS ARE NOW {total_winnings} DOLLARS.")
            # 打印当前总奖金
        else:
            # 如果没有猜对
            print(f"YOU BLEW IT...TOO BAD...THE NUMBER WAS {secret}")
            # 打印猜对的数字

        print("\n")
        # 打印空行
        print("PLAY AGAIN (YES OR NO)", end=QUESTION_PROMPT)
        # 打印提示
        answer = input().upper()
        # 获取用户输入并转换为大写
        if answer != "YES":
            # 如果回答不是 YES
            break
            # 退出循环

    print("\nSO LONG.  HOPE YOU ENJOYED YOURSELF!!!")
    # 打印结束语


if __name__ == "__main__":
    # 如果当前脚本被直接执行
    main()
    # 调用主函数

```