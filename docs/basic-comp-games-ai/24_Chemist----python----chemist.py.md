# `basic-computer-games\24_Chemist\python\chemist.py`

```py
"""
CHEMIST

A math game posing as a chemistry word problem.

Ported by Dave LeCompte
"""

import random

MAX_LIVES = 9  # 设置最大生命值为9


def play_scenario() -> bool:  # 定义一个函数，返回布尔值
    acid_amount = random.randint(1, 50)  # 生成1到50之间的随机整数作为acid_amount

    water_amount = 7 * acid_amount / 3  # 根据acid_amount计算water_amount

    print(f"{acid_amount} LITERS OF KRYPTOCYANIC ACID.  HOW MUCH WATER?")  # 打印提示信息

    response = float(input())  # 获取用户输入的响应

    difference = abs(water_amount - response)  # 计算水量与用户输入的差值的绝对值

    acceptable_difference = water_amount / 20  # 计算可接受的差值范围

    if difference > acceptable_difference:  # 判断差值是否超出可接受范围
        show_failure()  # 调用失败的提示函数
        return False  # 返回False
    else:
        show_success()  # 调用成功的提示函数
        return True  # 返回True


def show_failure() -> None:  # 定义一个函数，不返回任何值
    print(" SIZZLE!  YOU HAVE JUST BEEN DESALINATED INTO A BLOB")  # 打印失败的提示信息
    print(" OF QUIVERING PROTOPLASM!")


def show_success() -> None:  # 定义一个函数，不返回任何值
    print(" GOOD JOB! YOU MAY BREATHE NOW, BUT DON'T INHALE THE FUMES!\n")  # 打印成功的提示信息


def show_ending() -> None:  # 定义一个函数，不返回任何值
    print(f" YOUR {MAX_LIVES} LIVES ARE USED, BUT YOU WILL BE LONG REMEMBERED FOR")  # 打印游戏结束的提示信息
    print(" YOUR CONTRIBUTIONS TO THE FIELD OF COMIC BOOK CHEMISTRY.")


def main() -> None:  # 定义一个函数，不返回任何值
    print(" " * 33 + "CHEMIST")  # 打印游戏标题
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

    print("THE FICTITIOUS CHEMICAL KRYPTOCYANIC ACID CAN ONLY BE")  # 打印游戏规则
    print("DILUTED BY THE RATIO OF 7 PARTS WATER TO 3 PARTS ACID.")
    print("IF ANY OTHER RATIO IS ATTEMPTED, THE ACID BECOMES UNSTABLE")
    print("AND SOON EXPLODES.  GIVEN THE AMOUNT OF ACID, YOU MUST")
    print("DECIDE WHO MUCH WATER TO ADD FOR DILUTION.  IF YOU MISS")
    print("YOU FACE THE CONSEQUENCES.")

    lives_used = 0  # 初始化使用的生命值为0

    while True:  # 进入游戏循环
        success = play_scenario()  # 调用游戏场景函数

        if not success:  # 如果游戏场景失败
            lives_used += 1  # 使用的生命值加1

            if lives_used == MAX_LIVES:  # 如果使用的生命值等于最大生命值
                show_ending()  # 调用游戏结束的提示函数
                return  # 结束游戏


if __name__ == "__main__":  # 判断是否为主程序入口
    main()  # 调用主函数
```