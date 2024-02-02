# `basic-computer-games\51_Hurkle\python\hurkle.py`

```py
#!/usr/bin/env python3

"""Ported to Python by @iamtraction"""

from random import random


def direction(A, B, X, Y) -> None:
    """Print the direction hint for finding the hurkle."""

    # 打印寻找HURKLE的方向提示
    print("GO ", end="")
    if Y < B:
        print("NORTH", end="")
    elif Y > B:
        print("SOUTH", end="")

    if X < A:
        print("EAST", end="")
    elif X > A:
        print("WEST", end="")

    print()


def main() -> None:
    print(" " * 33 + "HURKLE")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")

    print("\n\n\n")

    N = 5
    G = 10

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

    while True:
        A = int(G * random())
        B = int(G * random())

        for k in range(0, N):
            print("\nGUESS #" + str(k))

            # 读取以`X, Y`格式的坐标，将字符串在`,`处分割，然后将坐标解析为`int`并分别存储在`X`和`Y`中。
            [X, Y] = [int(c) for c in input("X,Y? ").split(",")]

            if abs(X - A) + abs(Y - B) == 0:
                print("\nYOU FOUND HIM IN", k + 1, "GUESSES!")
                break
            else:
                direction(A, B, X, Y)
                continue

        print("\n\nLET'S PLAY AGAIN, HURKLE IS HIDING.\n")


if __name__ == "__main__":
    main()
```