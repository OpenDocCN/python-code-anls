# `basic-computer-games\24_Chemist\python\chemist.py`

```

"""
CHEMIST

A math game posing as a chemistry word problem.

Ported by Dave LeCompte
"""

import random  # 导入 random 模块

MAX_LIVES = 9  # 设置最大生命值为 9


def play_scenario() -> bool:  # 定义 play_scenario 函数，返回布尔值
    acid_amount = random.randint(1, 50)  # 生成 1 到 50 之间的随机整数作为 acid_amount

    water_amount = 7 * acid_amount / 3  # 根据 acid_amount 计算 water_amount

    print(f"{acid_amount} LITERS OF KRYPTOCYANIC ACID.  HOW MUCH WATER?")  # 打印提示信息

    response = float(input())  # 获取用户输入的响应

    difference = abs(water_amount - response)  # 计算水量与用户输入的差值的绝对值

    acceptable_difference = water_amount / 20  # 计算可接受的差值范围

    if difference > acceptable_difference:  # 判断差值是否超出可接受范围
        show_failure()  # 调用 show_failure 函数
        return False  # 返回 False
    else:
        show_success()  # 调用 show_success 函数
        return True  # 返回 True


def show_failure() -> None:  # 定义 show_failure 函数，不返回任何值
    print(" SIZZLE!  YOU HAVE JUST BEEN DESALINATED INTO A BLOB")  # 打印失败信息
    print(" OF QUIVERING PROTOPLASM!")


def show_success() -> None:  # 定义 show_success 函数，不返回任何值
    print(" GOOD JOB! YOU MAY BREATHE NOW, BUT DON'T INHALE THE FUMES!\n")  # 打印成功信息


def show_ending() -> None:  # 定义 show_ending 函数，不返回任何值
    print(f" YOUR {MAX_LIVES} LIVES ARE USED, BUT YOU WILL BE LONG REMEMBERED FOR")  # 打印结束信息
    print(" YOUR CONTRIBUTIONS TO THE FIELD OF COMIC BOOK CHEMISTRY.")


def main() -> None:  # 定义主函数，不返回任何值
    print(" " * 33 + "CHEMIST")  # 打印游戏标题
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")  # 打印创意计算信息

    # 打印游戏规则
    print("THE FICTITIOUS CHEMICAL KRYPTOCYANIC ACID CAN ONLY BE")
    print("DILUTED BY THE RATIO OF 7 PARTS WATER TO 3 PARTS ACID.")
    print("IF ANY OTHER RATIO IS ATTEMPTED, THE ACID BECOMES UNSTABLE")
    print("AND SOON EXPLODES.  GIVEN THE AMOUNT OF ACID, YOU MUST")
    print("DECIDE WHO MUCH WATER TO ADD FOR DILUTION.  IF YOU MISS")
    print("YOU FACE THE CONSEQUENCES.")

    lives_used = 0  # 初始化使用生命值的次数为 0

    while True:  # 进入游戏循环
        success = play_scenario()  # 调用 play_scenario 函数

        if not success:  # 如果失败
            lives_used += 1  # 生命值次数加一

            if lives_used == MAX_LIVES:  # 如果生命值次数达到最大值
                show_ending()  # 调用 show_ending 函数
                return  # 结束游戏


if __name__ == "__main__":  # 如果当前文件被直接运行
    main()  # 调用主函数

```