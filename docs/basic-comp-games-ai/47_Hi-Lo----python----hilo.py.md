# `basic-computer-games\47_Hi-Lo\python\hilo.py`

```
#!/usr/bin/env python3
# 设置 Python 脚本的解释器为 Python 3

import random
# 导入 random 模块

MAX_ATTEMPTS = 6
# 设置最大尝试次数为 6
QUESTION_PROMPT = "? "
# 设置问题提示符为 "?"

def main() -> None:
    # 主函数，不返回任何结果
    print("HI LO")
    # 打印 "HI LO"
    print("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    # 打印 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，包含多个换行
    print("THIS IS THE GAME OF HI LO.\n")
    # 打印 "THIS IS THE GAME OF HI LO."，包含一个换行
    print("YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE")
    # 打印 "YOU WILL HAVE 6 TRIES TO GUESS THE AMOUNT OF MONEY IN THE"
    print("HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU")
    # 打印 "HI LO JACKPOT, WHICH IS BETWEEN 1 AND 100 DOLLARS.  IF YOU"
    print("GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!")
    # 打印 "GUESS THE AMOUNT, YOU WIN ALL THE MONEY IN THE JACKPOT!"
    print("THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,")
    # 打印 "THEN YOU GET ANOTHER CHANCE TO WIN MORE MONEY.  HOWEVER,"
    print("IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS.\n\n")
    # 打印 "IF YOU DO NOT GUESS THE AMOUNT, THE GAME ENDS."，包含两个换行

    total_winnings = 0
    # 初始化总奖金为 0
    while True:
        # 进入无限循环
        print()
        # 打印一个空行
        secret = random.randint(1, 100)
        # 生成一个 1 到 100 之间的随机数作为秘密数字
        guessed_correctly = False
        # 初始化猜对标志为 False

        for _attempt in range(MAX_ATTEMPTS):
            # 循环最大尝试次数次
            print("YOUR GUESS", end=QUESTION_PROMPT)
            # 打印 "YOUR GUESS"，并以问题提示符结尾
            guess = int(input())
            # 获取用户输入的猜测并转换为整数

            if guess == secret:
                # 如果猜测等于秘密数字
                print(f"GOT IT!!!!!!!!!!   YOU WIN {secret} DOLLARS.")
                # 打印 "GOT IT!!!!!!!!!!   YOU WIN {secret} DOLLARS."
                guessed_correctly = True
                # 设置猜对标志为 True
                break
                # 跳出循环
            elif guess > secret:
                # 如果猜测大于秘密数字
                print("YOUR GUESS IS TOO HIGH.")
                # 打印 "YOUR GUESS IS TOO HIGH."
            else:
                # 如果猜测小于秘密数字
                print("YOUR GUESS IS TOO LOW.")
                # 打印 "YOUR GUESS IS TOO LOW."

        if guessed_correctly:
            # 如果猜对标志为 True
            total_winnings += secret
            # 总奖金增加秘密数字的金额
            print(f"YOUR TOTAL WINNINGS ARE NOW {total_winnings} DOLLARS.")
            # 打印当前总奖金
        else:
            # 如果猜对标志为 False
            print(f"YOU BLEW IT...TOO BAD...THE NUMBER WAS {secret}")
            # 打印 "YOU BLEW IT...TOO BAD...THE NUMBER WAS {secret}"

        print("\n")
        # 打印一个空行
        print("PLAY AGAIN (YES OR NO)", end=QUESTION_PROMPT)
        # 打印 "PLAY AGAIN (YES OR NO)"，并以问题提示符结尾
        answer = input().upper()
        # 获取用户输入的答案并转换为大写
        if answer != "YES":
            # 如果答案不是 "YES"
            break
            # 退出循环

    print("\nSO LONG.  HOPE YOU ENJOYED YOURSELF!!!")
    # 打印 "SO LONG.  HOPE YOU ENJOYED YOURSELF!!!"

if __name__ == "__main__":
    # 如果当前脚本被直接执行
    main()
    # 调用主函数
```