# `basic-computer-games\51_Hurkle\python\hurkle.py`

```

#!/usr/bin/env python3
# 指定脚本解释器为 Python3

"""Ported to Python by @iamtraction"""
# 添加作者注释

from random import random
# 导入 random 模块的 random 函数

def direction(A, B, X, Y) -> None:
    """Print the direction hint for finding the hurkle."""
    # 打印寻找 hurkle 的方向提示

    print("GO ", end="")
    # 打印 "GO "，不换行
    if Y < B:
        print("NORTH", end="")
    # 如果 Y 坐标小于 B，则打印 "NORTH"，不换行
    elif Y > B:
        print("SOUTH", end="")
    # 如果 Y 坐标大于 B，则打印 "SOUTH"，不换行

    if X < A:
        print("EAST", end="")
    # 如果 X 坐标小于 A，则打印 "EAST"，不换行
    elif X > A:
        print("WEST", end="")
    # 如果 X 坐标大于 A，则打印 "WEST"，不换行

    print()
    # 换行


def main() -> None:
    print(" " * 33 + "HURKLE")
    # 打印空格 33 个和 "HURKLE"
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    # 打印空格 15 个和 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"

    print("\n\n\n")
    # 打印三个空行

    N = 5
    G = 10
    # 初始化变量 N 和 G

    print()
    print("A HURKLE IS HIDING ON A", G, "BY", G, "GRID. HOMEBASE")
    print("ON THE GRID IS POINT 0,0 IN THE SOUTHWEST CORNER,")
    print("AND ANY POINT ON THE GRID IS DESIGNATED BY A")
    print("PAIR OF WHOLE NUMBERS SEPERATED BY A COMMA. THE FIRST")
    print("NUMBER IS THE HORIZONTAL POSITION AND THE SECOND NUMBER")
    print("IS THE VERTICAL POSITION. YOU MUST TRY TO")
    print("GUESS THE HURKLE'S GRIDPOINT. YOU GET", N, "TRIES.")
    print("AFTER EACH TRY, I WILL TELL YOU THE APPROXIMATE")
    print("DIRECTION TO GO TO LOOK FOR THE HURKLE.")
    print()
    # 打印游戏说明

    while True:
        A = int(G * random())
        B = int(G * random())
        # 生成随机的 A 和 B 坐标

        for k in range(0, N):
            print("\nGUESS #" + str(k))
            # 打印猜测次数

            [X, Y] = [int(c) for c in input("X,Y? ").split(",")]
            # 读取以 `X, Y` 格式输入的坐标，使用逗号分割字符串，然后将坐标解析为整数并分别存储在 `X` 和 `Y` 中

            if abs(X - A) + abs(Y - B) == 0:
                print("\nYOU FOUND HIM IN", k + 1, "GUESSES!")
                break
            else:
                direction(A, B, X, Y)
                continue
            # 如果猜中了 hurkle，则打印提示信息并结束游戏；否则打印方向提示并继续游戏

        print("\n\nLET'S PLAY AGAIN, HURKLE IS HIDING.\n")
        # 打印再玩一次的提示


if __name__ == "__main__":
    main()
# 如果当前脚本为主程序，则执行 main 函数

```