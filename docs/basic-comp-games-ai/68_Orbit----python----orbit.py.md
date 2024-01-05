# `68_Orbit\python\orbit.py`

```
"""
ORBIT

Orbital mechanics simulation

Port by Dave LeCompte
"""

import math  # 导入数学库
import random  # 导入随机数库

PAGE_WIDTH = 64  # 设置页面宽度为64

# 定义一个函数，用于打印居中的消息
def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)  # 计算需要添加的空格数，使得消息居中
    print(spaces + msg)  # 打印居中的消息

# 定义一个函数，用于打印说明
def print_instructions() -> None:
    print(
        """SOMEWHERE ABOVE YOUR PLANET IS A ROMULAN SHIP.

THE SHIP IS IN A CONSTANT POLAR ORBIT.  ITS
DISTANCE FROM THE CENTER OF YOUR PLANET IS FROM
10,000 TO 30,000 MILES AND AT ITS PRESENT VELOCITY CAN
CIRCLE YOUR PLANET ONCE EVERY 12 TO 36 HOURS.

UNFORTUNATELY, THEY ARE USING A CLOAKING DEVICE SO
YOU ARE UNABLE TO SEE THEM, BUT WITH A SPECIAL
INSTRUMENT YOU CAN TELL HOW NEAR THEIR SHIP YOUR
PHOTON BOMB EXPLODED.  YOU HAVE SEVEN HOURS UNTIL THEY
HAVE BUILT UP SUFFICIENT POWER IN ORDER TO ESCAPE
YOUR PLANET'S GRAVITY.

YOUR PLANET HAS ENOUGH POWER TO FIRE ONE BOMB AN HOUR.

AT THE BEGINNING OF EACH HOUR YOU WILL BE ASKED TO GIVE AN
ANGLE (BETWEEN 0 AND 360) AND A DISTANCE IN UNITS OF
100 MILES (BETWEEN 100 AND 300), AFTER WHICH YOUR BOMB'S
```
这段代码是一个多行的打印语句，用于输出一段描述性的文本。
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，并封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象的文件名列表，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
def get_yes_or_no() -> bool:
    # 无限循环，直到用户输入了"YES"或者"NO"
    while True:
        # 获取用户输入并转换为大写
        response = input().upper()
        # 如果用户输入了"YES"，返回True
        if response == "YES":
            return True
        # 如果用户输入了"NO"，返回False
        elif response == "NO":
            return False
        # 如果用户输入了其他内容，提示用户重新输入
        else:
            print("PLEASE TYPE 'YES' OR 'NO'")


def game_over(is_success: bool) -> bool:
    # 如果任务成功，打印成功信息
    if is_success:
        print("YOU HAVE SUCCESSFULLY COMPLETED YOUR MISSION.")
    # 如果任务失败，打印失败信息
    else:
        print("YOU HAVE ALLOWED THE ROMULANS TO ESCAPE.")
    # 打印另一艘罗穆兰船进入轨道的信息
    print("ANOTHER ROMULAN SHIP HAS GONE INTO ORBIT.")
    print("DO YOU WISH TO TRY TO DESTROY IT?")
    # 打印提示信息，询问用户是否想要尝试摧毁目标

    return get_yes_or_no()
    # 调用函数 get_yes_or_no()，并返回其结果


def play_game() -> bool:
    rom_angle = random.randint(0, 359)
    # 生成一个随机的角度值，范围在 0 到 359 之间
    rom_distance = random.randint(100, 300)
    # 生成一个随机的距离值，范围在 100 到 300 之间
    rom_angular_velocity = random.randint(10, 30)
    # 生成一个随机的角速度值，范围在 10 到 30 之间
    hour = 0
    # 初始化小时数为 0
    while hour < 7:
        # 当小时数小于 7 时执行循环
        hour += 1
        # 小时数加一
        print()
        # 打印空行
        print()
        # 打印空行
        print(f"THIS IS HOUR {hour}, AT WHAT ANGLE DO YOU WISH TO SEND")
        # 打印当前小时数，并询问用户希望发送光子炸弹的角度
        print("YOUR PHOTON BOMB?")
        # 打印提示信息，询问用户希望发送光子炸弹的角度

        bomb_angle = float(input())
        # 获取用户输入的光子炸弹角度，并转换为浮点数
        print("HOW FAR OUT DO YOU WISH TO DETONATE IT?")
        # 打印提示信息，询问用户希望在多远的地方引爆光子炸弹
        bomb_distance = float(input())
        # 获取用户输入的光子炸弹引爆距离，并转换为浮点数
        # 打印空行
        print()
        # 打印空行
        print()

        # 计算 ROM 飞船的角度
        rom_angle = (rom_angle + rom_angular_velocity) % 360
        # 计算 ROM 飞船角度与炸弹角度的差值
        angular_difference = rom_angle - bomb_angle
        # 计算炸弹与 ROM 飞船之间的距离
        c = math.sqrt(
            rom_distance**2
            + bomb_distance**2
            - 2
            * rom_distance
            * bomb_distance
            * math.cos(math.radians(angular_difference))
        )

        # 打印炸弹爆炸的位置
        print(f"YOUR PHOTON BOMB EXPLODED {c:.4f}*10^2 MILES FROM THE")
        print("ROMULAN SHIP.")

        # 如果炸弹爆炸距离小于等于50，返回 True 表示摧毁了 Romulan 飞船
        if c <= 50:
            # Destroyed the Romulan
            return True
    # Ran out of time
    return False  # 如果程序运行超时，则返回 False

def main() -> None:
    print_centered("ORBIT")  # 打印居中的标题 "ORBIT"
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")  # 打印居中的文本 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n"

    print_instructions()  # 打印游戏的说明

    while True:  # 进入游戏循环
        success = play_game()  # 进行游戏，返回游戏是否成功的标志
        again = game_over(success)  # 根据游戏是否成功的标志，打印游戏结束信息，并询问是否再玩一次
        if not again:  # 如果不再玩一次，则结束游戏循环
            return

if __name__ == "__main__":
    main()  # 调用主函数开始游戏
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```