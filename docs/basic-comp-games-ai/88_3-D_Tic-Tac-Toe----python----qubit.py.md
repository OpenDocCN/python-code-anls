# `basic-computer-games\88_3-D_Tic-Tac-Toe\python\qubit.py`

```

#!/usr/bin/env python3
# 指定脚本解释器为 Python3

# 从 BASIC Computer Games 中的 3D Tic Tac Toe 的 BASIC 源代码移植而来
# 代码来源于 Dartmouth College

from enum import Enum
from typing import Optional, Tuple, Union

# 定义移动的枚举类型
class Move(Enum):
    """Game status and types of machine move"""
    HUMAN_WIN = 0
    MACHINE_WIN = 1
    DRAW = 2
    MOVES = 3
    LIKES = 4
    TAKES = 5
    GET_OUT = 6
    YOU_FOX = 7
    NICE_TRY = 8
    CONCEDES = 9

# 定义玩家的枚举类型
class Player(Enum):
    EMPTY = 0
    HUMAN = 1
    MACHINE = 2

# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 创建 Qubit 对象
    game = Qubit()
    # 开始游戏
    game.play()

```