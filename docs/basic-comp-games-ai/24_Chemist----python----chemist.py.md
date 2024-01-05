# `24_Chemist\python\chemist.py`

```
"""
CHEMIST

A math game posing as a chemistry word problem.

Ported by Dave LeCompte
"""

import random  # 导入 random 模块，用于生成随机数

MAX_LIVES = 9  # 设置最大生命值为 9


def play_scenario() -> bool:  # 定义一个名为 play_scenario 的函数，返回布尔值
    acid_amount = random.randint(1, 50)  # 生成 1 到 50 之间的随机整数，表示酸的数量

    water_amount = 7 * acid_amount / 3  # 计算水的数量，为酸的数量的 7/3 倍

    print(f"{acid_amount} LITERS OF KRYPTOCYANIC ACID.  HOW MUCH WATER?")  # 打印酸的数量和提示信息
    response = float(input())  # 从用户输入中获取浮点数作为响应值

    difference = abs(water_amount - response)  # 计算水量和响应值之间的差值

    acceptable_difference = water_amount / 20  # 计算水量的可接受差值范围

    if difference > acceptable_difference:  # 如果差值大于可接受范围
        show_failure()  # 调用显示失败的函数
        return False  # 返回 False 表示失败
    else:  # 如果差值在可接受范围内
        show_success()  # 调用显示成功的函数
        return True  # 返回 True 表示成功


def show_failure() -> None:  # 定义显示失败的函数，不返回任何值
    print(" SIZZLE!  YOU HAVE JUST BEEN DESALINATED INTO A BLOB")  # 打印失败消息
    print(" OF QUIVERING PROTOPLASM!")  # 打印失败消息
def show_success() -> None:
    # 打印成功消息
    print(" GOOD JOB! YOU MAY BREATHE NOW, BUT DON'T INHALE THE FUMES!\n")


def show_ending() -> None:
    # 打印游戏结束消息
    print(f" YOUR {MAX_LIVES} LIVES ARE USED, BUT YOU WILL BE LONG REMEMBERED FOR")
    print(" YOUR CONTRIBUTIONS TO THE FIELD OF COMIC BOOK CHEMISTRY.")


def main() -> None:
    # 打印游戏标题
    print(" " * 33 + "CHEMIST")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

    # 打印游戏背景信息
    print("THE FICTITIOUS CHEMICAL KRYPTOCYANIC ACID CAN ONLY BE")
    print("DILUTED BY THE RATIO OF 7 PARTS WATER TO 3 PARTS ACID.")
    print("IF ANY OTHER RATIO IS ATTEMPTED, THE ACID BECOMES UNSTABLE")
    print("AND SOON EXPLODES.  GIVEN THE AMOUNT OF ACID, YOU MUST")
    print("DECIDE WHO MUCH WATER TO ADD FOR DILUTION.  IF YOU MISS")
    print("YOU FACE THE CONSEQUENCES.")
    lives_used = 0  # 初始化使用的生命次数为0

    while True:  # 进入无限循环
        success = play_scenario()  # 调用play_scenario()函数，返回成功与否的标志

        if not success:  # 如果不成功
            lives_used += 1  # 使用的生命次数加1

            if lives_used == MAX_LIVES:  # 如果使用的生命次数达到最大生命次数
                show_ending()  # 调用show_ending()函数显示结局
                return  # 结束循环

if __name__ == "__main__":  # 如果当前脚本被直接执行
    main()  # 调用main()函数
```