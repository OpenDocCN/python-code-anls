# `basic-computer-games\42_Gunner\python\gunner.py`

```
#!/usr/bin/env python3
#
# Ported to Python by @iamtraction

# 从 math 模块中导入 sin 函数
from math import sin
# 从 random 模块中导入 random 函数
from random import random

# 定义 gunner 函数，无返回值
def gunner() -> None:
    # 计算枪的最大射程
    gun_range = int(40000 * random() + 20000)

    # 打印枪的最大射程
    print("\nMAXIMUM RANGE OF YOUR GUN IS", gun_range, "YARDS.")

    # 初始化击毙敌人数和 S1 变量
    killed_enemies = 0
    S1 = 0

# 定义 main 函数，无返回值
def main() -> None:
    # 打印游戏标题
    print(" " * 33 + "GUNNER")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n\n\n")
    print("YOU ARE THE OFFICER-IN-CHARGE, GIVING ORDERS TO A GUN")
    print("CREW, TELLING THEM THE DEGREES OF ELEVATION YOU ESTIMATE")
    print("WILL PLACE A PROJECTILE ON TARGET.  A HIT WITHIN 100 YARDS")
    print("OF THE TARGET WILL DESTROY IT.")

    # 循环进行游戏
    while True:
        gunner()

        # 询问是否再次进行游戏
        not_again = input("TRY AGAIN (Y OR N)? ").upper() != "Y"
        if not_again:
            print("\nOK.  RETURN TO BASE CAMP.")
            break

# 如果当前脚本被直接执行，则调用 main 函数
if __name__ == "__main__":
    main()
```