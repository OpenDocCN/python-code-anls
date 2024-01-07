# `basic-computer-games\09_Battle\python\battle.py`

```

#!/usr/bin/env python3
# 指定使用 Python3 解释器

from random import randrange
from typing import List, Tuple
# 导入需要使用的模块和类型

PointType = Tuple[int, int]
VectorType = PointType
SeaType = Tuple[List[int], ...]
# 定义类型别名

SEA_WIDTH = 6
DESTROYER_LENGTH = 2
CRUISER_LENGTH = 3
AIRCRAFT_CARRIER_LENGTH = 4
# 定义常量

def random_vector() -> Tuple[int, int]:
    # 生成随机向量
    while True:
        vector = (randrange(-1, 2), randrange(-1, 2))

        if vector == (0, 0):
            # 如果向量为零向量，则重新生成
            continue

        return vector

def add_vector(point: PointType, vector: VectorType) -> PointType:
    # 向点添加向量
    return (point[0] + vector[0], point[1] + vector[1])

def place_ship(sea: SeaType, size: int, code: int) -> None:
    # 在海域中放置船只
    while True:
        start = (randrange(1, SEA_WIDTH + 1), randrange(1, SEA_WIDTH + 1))
        vector = random_vector()

        # 获取潜在的船只点
        point = start
        points = []

        for _ in range(size):
            point = add_vector(point, vector)
            points.append(point)

        if not all([is_within_sea(point, sea) for point in points]) or any(
            [value_at(point, sea) for point in points]
        ):
            # 如果船只超出边界或与其他船只交叉，则重新放置
            continue

        # 找到有效位置，放置船只
        for point in points:
            set_value_at(code, point, sea)

        break

def print_encoded_sea(sea: SeaType) -> None:
    # 打印加密的海域
    for x in range(len(sea)):
        print(" ".join([str(sea[y][x]) for y in range(len(sea) - 1, -1, -1)]))

# 其余函数的作用和注释与示例中的函数类似，不再赘述

```