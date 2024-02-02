# `basic-computer-games\68_Orbit\python\orbit.py`

```py
# 定义了一个多行字符串，用于描述游戏的背景和玩法
"""
ORBIT

Orbital mechanics simulation

Port by Dave LeCompte
"""

# 导入 math 和 random 模块
import math
import random

# 定义页面宽度常量
PAGE_WIDTH = 64


# 定义一个函数，用于打印居中的消息
def print_centered(msg: str) -> None:
    # 计算需要添加的空格数，使得消息居中显示
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    # 打印居中的消息
    print(spaces + msg)


# 定义一个函数，用于打印游戏说明
def print_instructions() -> None:
    # 打印游戏说明的多行字符串
    print(
        """SOMEWHERE ABOVE YOUR PLANET IS A ROMULAN SHIP.
        
        ...（以下省略，打印游戏说明的其余部分）
        """
# 定义一个函数，用于获取用户输入的是或否，返回布尔值
def get_yes_or_no() -> bool:
    while True:
        response = input().upper()  # 获取用户输入并转换为大写
        if response == "YES":  # 如果用户输入是YES，则返回True
            return True
        elif response == "NO":  # 如果用户输入是NO，则返回False
            return False
        else:  # 如果用户输入既不是YES也不是NO，则提示用户重新输入
            print("PLEASE TYPE 'YES' OR 'NO'")

# 定义一个函数，用于展示游戏结束的信息，并根据用户选择是否继续游戏
def game_over(is_success: bool) -> bool:
    if is_success:  # 如果游戏成功，打印成功信息
        print("YOU HAVE SUCCESSFULLY COMPLETED YOUR MISSION.")
    else:  # 如果游戏失败，打印失败信息
        print("YOU HAVE ALLOWED THE ROMULANS TO ESCAPE.")
    print("ANOTHER ROMULAN SHIP HAS GONE INTO ORBIT.")
    print("DO YOU WISH TO TRY TO DESTROY IT?")

    return get_yes_or_no()  # 调用获取用户输入的函数，返回用户是否继续游戏的选择

# 定义一个函数，用于开始游戏并初始化罗穆兰飞船的角度、距离和角速度
def play_game() -> bool:
    rom_angle = random.randint(0, 359)  # 随机生成罗穆兰飞船的角度
    rom_distance = random.randint(100, 300)  # 随机生成罗穆兰飞船的距离
    rom_angular_velocity = random.randint(10, 30)  # 随机生成罗穆兰飞船的角速度
    hour = 0  # 初始化游戏时间为0小时
    # 当小时数小于7时执行循环
    while hour < 7:
        # 小时数加1
        hour += 1
        # 打印空行
        print()
        print()
        # 打印当前小时数和提示信息
        print(f"THIS IS HOUR {hour}, AT WHAT ANGLE DO YOU WISH TO SEND")
        print("YOUR PHOTON BOMB?")
        
        # 获取用户输入的炸弹角度和爆炸距离
        bomb_angle = float(input())
        print("HOW FAR OUT DO YOU WISH TO DETONATE IT?")
        bomb_distance = float(input())
        print()
        print()

        # 计算罗穆兰船的角度
        rom_angle = (rom_angle + rom_angular_velocity) % 360
        # 计算角度差
        angular_difference = rom_angle - bomb_angle
        # 计算爆炸位置与罗穆兰船的距离
        c = math.sqrt(
            rom_distance**2
            + bomb_distance**2
            - 2
            * rom_distance
            * bomb_distance
            * math.cos(math.radians(angular_difference))
        )

        # 打印爆炸位置信息
        print(f"YOUR PHOTON BOMB EXPLODED {c:.4f}*10^2 MILES FROM THE")
        print("ROMULAN SHIP.")

        # 如果爆炸距离小于等于50，则摧毁罗穆兰船，返回True
        if c <= 50:
            # Destroyed the Romulan
            return True

    # 如果小时数超过7，则返回False，表示时间用尽
    # Ran out of time
    return False
# 定义主函数，不返回任何结果
def main() -> None:
    # 居中打印标题
    print_centered("ORBIT")
    # 居中打印副标题
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")

    # 打印游戏说明
    print_instructions()

    # 循环进行游戏
    while True:
        # 进行游戏并返回是否成功
        success = play_game()
        # 根据游戏结果判断是否重新开始游戏
        again = game_over(success)
        # 如果不需要重新开始游戏，则结束主函数
        if not again:
            return


# 如果当前脚本为主程序，则执行主函数
if __name__ == "__main__":
    main()
```