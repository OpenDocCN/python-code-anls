# `basic-computer-games\68_Orbit\python\orbit.py`

```

"""
ORBIT

Orbital mechanics simulation

Port by Dave LeCompte
"""

import math  # 导入数学库
import random  # 导入随机数库

PAGE_WIDTH = 64  # 定义页面宽度为64


def print_centered(msg: str) -> None:  # 定义一个函数，用于打印居中的消息
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)  # 计算需要添加的空格数
    print(spaces + msg)  # 打印居中的消息


def print_instructions() -> None:  # 定义一个函数，用于打印游戏说明
    print(
        """SOMEWHERE ABOVE YOUR PLANET IS A ROMULAN SHIP.
        ... (游戏说明)
        GOOD LUCK.  THE FEDERATION IS COUNTING ON YOU.
    """
    )


def get_yes_or_no() -> bool:  # 定义一个函数，用于获取用户输入的是或否
    while True:
        response = input().upper()  # 获取用户输入并转换为大写
        if response == "YES":
            return True
        elif response == "NO":
            return False
        else:
            print("PLEASE TYPE 'YES' OR 'NO'")  # 提示用户输入错误


def game_over(is_success: bool) -> bool:  # 定义一个函数，用于游戏结束时的处理
    if is_success:
        print("YOU HAVE SUCCESSFULLY COMPLETED YOUR MISSION.")  # 打印成功消息
    else:
        print("YOU HAVE ALLOWED THE ROMULANS TO ESCAPE.")  # 打印失败消息
    print("ANOTHER ROMULAN SHIP HAS GONE INTO ORBIT.")  # 提示另一艘敌舰进入轨道
    print("DO YOU WISH TO TRY TO DESTROY IT?")  # 询问用户是否继续游戏

    return get_yes_or_no()  # 获取用户输入的是或否


def play_game() -> bool:  # 定义一个函数，用于进行游戏
    rom_angle = random.randint(0, 359)  # 随机生成敌舰的角度
    rom_distance = random.randint(100, 300)  # 随机生成敌舰的距离
    rom_angular_velocity = random.randint(10, 30)  # 随机生成敌舰的角速度
    hour = 0  # 初始化小时数
    while hour < 7:  # 当小时数小于7时循环
        hour += 1  # 小时数加1
        print()
        print()
        print(f"THIS IS HOUR {hour}, AT WHAT ANGLE DO YOU WISH TO SEND")  # 打印当前小时数和询问用户发送炸弹的角度
        print("YOUR PHOTON BOMB?")

        bomb_angle = float(input())  # 获取用户输入的炸弹角度
        print("HOW FAR OUT DO YOU WISH TO DETONATE IT?")  # 询问用户炸弹爆炸距离
        bomb_distance = float(input())  # 获取用户输入的炸弹爆炸距离
        print()
        print()

        rom_angle = (rom_angle + rom_angular_velocity) % 360  # 更新敌舰的角度
        angular_difference = rom_angle - bomb_angle  # 计算角度差
        c = math.sqrt(  # 计算炸弹爆炸距离
            rom_distance**2
            + bomb_distance**2
            - 2
            * rom_distance
            * bomb_distance
            * math.cos(math.radians(angular_difference))
        )

        print(f"YOUR PHOTON BOMB EXPLODED {c:.4f}*10^2 MILES FROM THE")  # 打印炸弹爆炸距离
        print("ROMULAN SHIP.")

        if c <= 50:  # 如果炸弹爆炸距离小于等于50
            # Destroyed the Romulan
            return True  # 返回游戏成功

    # Ran out of time
    return False  # 返回游戏失败


def main() -> None:  # 定义主函数
    print_centered("ORBIT")  # 打印游戏标题
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")  # 打印创意计算公司信息

    print_instructions()  # 打印游戏说明

    while True:  # 无限循环
        success = play_game()  # 进行游戏
        again = game_over(success)  # 游戏结束处理
        if not again:  # 如果用户选择不再玩游戏
            return  # 结束游戏


if __name__ == "__main__":  # 如果当前脚本为主程序
    main()  # 调用主函数

```