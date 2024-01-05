# `51_Hurkle\python\hurkle.py`

```
#!/usr/bin/env python3
# 指定脚本的解释器为 Python 3

"""Ported to Python by @iamtraction"""
# 添加脚本的作者信息

from random import random
# 从 random 模块中导入 random 函数

def direction(A, B, X, Y) -> None:
    """Print the direction hint for finding the hurkle."""
    # 定义一个函数，用于打印寻找 hurkle 的方向提示

    print("GO ", end="")
    # 打印 "GO "，并且不换行

    if Y < B:
        print("NORTH", end="")
    # 如果 Y 坐标小于 B 坐标，打印 "NORTH"，并且不换行
    elif Y > B:
        print("SOUTH", end="")
    # 如果 Y 坐标大于 B 坐标，打印 "SOUTH"，并且不换行

    if X < A:
        print("EAST", end="")
    # 如果 X 坐标小于 A 坐标，打印 "EAST"，并且不换行
    elif X > A:
        print("WEST", end="")
    # 如果 X 坐标大于 A 坐标，打印 "WEST"，并且不换行
    print()
    # 定义一个函数，没有参数，没有返回值
def main() -> None:
    # 打印"HURKLE"，并在前面加上33个空格
    print(" " * 33 + "HURKLE")
    # 打印"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并在前面加上15个空格
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")

    # 打印三个空行
    print("\n\n\n")

    # 定义变量N并赋值为5，定义变量G并赋值为10
    N = 5
    G = 10

    # 打印空行
    print()
    # 打印"A HURKLE IS HIDING ON A", G, "BY", G, "GRID. HOMEBASE"
    print("A HURKLE IS HIDING ON A", G, "BY", G, "GRID. HOMEBASE")
    # 打印"ON THE GRID IS POINT 0,0 IN THE SOUTHWEST CORNER,"
    print("ON THE GRID IS POINT 0,0 IN THE SOUTHWEST CORNER,")
    # 打印"AND ANY POINT ON THE GRID IS DESIGNATED BY A"
    print("AND ANY POINT ON THE GRID IS DESIGNATED BY A")
    # 打印"PAIR OF WHOLE NUMBERS SEPERATED BY A COMMA. THE FIRST"
    print("PAIR OF WHOLE NUMBERS SEPERATED BY A COMMA. THE FIRST")
    # 打印"NUMBER IS THE HORIZONTAL POSITION AND THE SECOND NUMBER"
    print("NUMBER IS THE HORIZONTAL POSITION AND THE SECOND NUMBER")
    # 打印"IS THE VERTICAL POSITION. YOU MUST TRY TO"
    print("IS THE VERTICAL POSITION. YOU MUST TRY TO")
    # 打印猜测的次数和提示信息
    print("GUESS THE HURKLE'S GRIDPOINT. YOU GET", N, "TRIES.")
    print("AFTER EACH TRY, I WILL TELL YOU THE APPROXIMATE")
    print("DIRECTION TO GO TO LOOK FOR THE HURKLE.")
    print()

    # 无限循环，直到找到HURKLE或者用户选择退出
    while True:
        # 生成随机的A和B坐标
        A = int(G * random())
        B = int(G * random())

        # 循环N次，进行猜测
        for k in range(0, N):
            # 打印当前猜测的次数
            print("\nGUESS #" + str(k))

            # 读取用户输入的坐标，并将其解析为整数类型的X和Y
            [X, Y] = [int(c) for c in input("X,Y? ").split(",")]

            # 如果猜测的坐标与HURKLE的坐标相同，则打印找到HURKLE的信息并跳出循环
            if abs(X - A) + abs(Y - B) == 0:
                print("\nYOU FOUND HIM IN", k + 1, "GUESSES!")
                break
            else:  # 如果不是找到了 Hurkle 的位置
                direction(A, B, X, Y)  # 调用 direction 函数，传入 A, B, X, Y 参数
                continue  # 继续下一轮循环

        print("\n\nLET'S PLAY AGAIN, HURKLE IS HIDING.\n")  # 打印提示信息，表示游戏重新开始

if __name__ == "__main__":  # 如果当前脚本被直接执行
    main()  # 调用 main 函数
```