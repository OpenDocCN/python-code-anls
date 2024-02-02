# `basic-computer-games\09_Battle\python\battle.py`

```py
#!/usr/bin/env python3
from random import randrange
from typing import List, Tuple

PointType = Tuple[int, int]  # 定义元组类型 PointType
VectorType = PointType  # 定义向量类型 VectorType
SeaType = Tuple[List[int], ...]  # 定义海域类型 SeaType

SEA_WIDTH = 6  # 定义海域宽度
DESTROYER_LENGTH = 2  # 驱逐舰长度
CRUISER_LENGTH = 3  # 巡洋舰长度
AIRCRAFT_CARRIER_LENGTH = 4  # 航空母舰长度


def random_vector() -> Tuple[int, int]:  # 生成随机向量
    while True:
        vector = (randrange(-1, 2), randrange(-1, 2))  # 生成随机向量

        if vector == (0, 0):
            # We can't have a zero vector, so try again
            continue

        return vector


def add_vector(point: PointType, vector: VectorType) -> PointType:  # 向点添加向量
    return (point[0] + vector[0], point[1] + vector[1])  # 返回新的点


def place_ship(sea: SeaType, size: int, code: int) -> None:  # 在海域中放置船只
    while True:
        start = (randrange(1, SEA_WIDTH + 1), randrange(1, SEA_WIDTH + 1))  # 随机生成起始点
        vector = random_vector()  # 生成随机向量

        # Get potential ship points
        point = start
        points = []

        for _ in range(size):
            point = add_vector(point, vector)  # 根据向量计算下一个点
            points.append(point)  # 将点添加到列表中

        if not all([is_within_sea(point, sea) for point in points]) or any(
            [value_at(point, sea) for point in points]
        ):
            # ship out of bounds or crosses other ship, trying again
            continue

        # We found a valid spot, so actually place it now
        for point in points:
            set_value_at(code, point, sea)  # 在海域中放置船只

        break


def print_encoded_sea(sea: SeaType) -> None:  # 打印编码后的海域
    for x in range(len(sea)):
        print(" ".join([str(sea[y][x]) for y in range(len(sea) - 1, -1, -1)]))  # 打印编码后的海域


def is_within_sea(point: PointType, sea: SeaType) -> bool:  # 判断点是否在海域内
    return (1 <= point[0] <= len(sea)) and (1 <= point[1] <= len(sea))  # 判断点是否在海域内


def has_ship(sea: SeaType, code: int) -> bool:  # 判断海域中是否有指定编码的船只
    return any(code in row for row in sea)  # 判断海域中是否有指定编码的船只


def count_sunk(sea: SeaType, *codes: int) -> int:  # 计算沉没的船只数量
    return sum(not has_ship(sea, code) for code in codes)  # 统计海域中指定编码的船只数量


def value_at(point: PointType, sea: SeaType) -> int:  # 获取指定点的值
    return sea[point[1] - 1][point[0] - 1]  # 获取指定点的值
# 在海域中的指定位置设置数值
def set_value_at(value: int, point: PointType, sea: SeaType) -> None:
    sea[point[1] - 1][point[0] - 1] = value

# 获取下一个目标位置
def get_next_target(sea: SeaType) -> PointType:
    while True:
        try:
            # 从用户输入中获取猜测的目标位置
            guess = input("? ")
            point_str_list = guess.split(",")

            # 如果输入不是两个数字，抛出值错误
            if len(point_str_list) != 2:
                raise ValueError()

            # 将输入的字符串转换为坐标点
            point = (int(point_str_list[0]), int(point_str_list[1]))

            # 如果坐标点不在海域范围内，抛出值错误
            if not is_within_sea(point, sea):
                raise ValueError()

            # 返回有效的目标位置
            return point
        except ValueError:
            # 捕获值错误并提示用户重新输入
            print(
                f"INVALID. SPECIFY TWO NUMBERS FROM 1 TO {len(sea)}, SEPARATED BY A COMMA."
            )

# 在海域中设置船只的位置
def setup_ships(sea: SeaType) -> None:
    place_ship(sea, DESTROYER_LENGTH, 1)
    place_ship(sea, DESTROYER_LENGTH, 2)
    place_ship(sea, CRUISER_LENGTH, 3)
    place_ship(sea, CRUISER_LENGTH, 4)
    place_ship(sea, AIRCRAFT_CARRIER_LENGTH, 5)
    place_ship(sea, AIRCRAFT_CARRIER_LENGTH, 6)

# 主函数
def main() -> None:
    # 创建一个海域，初始化为0
    sea = tuple([0 for _ in range(SEA_WIDTH)] for _ in range(SEA_WIDTH))
    # 在海域中设置船只的位置
    setup_ships(sea)
    # 打印加密的海域信息
    print(
        """
                BATTLE
CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY

THE FOLLOWING CODE OF THE BAD GUYS' FLEET DISPOSITION
HAS BEEN CAPTURED BUT NOT DECODED:

"""
    )
    print_encoded_sea(sea)
    # 打印游戏开始提示
    print(
        """

DE-CODE IT AND USE IT IF YOU CAN
BUT KEEP THE DE-CODING METHOD A SECRET.

START GAME"""
    )
    # 初始化击中和未击中的次数
    splashes = 0
    hits = 0
    # 无限循环，直到条件被打破
    while True:
        # 获取下一个目标的位置
        target = get_next_target(sea)
        # 获取目标位置的值
        target_value = value_at(target, sea)

        # 如果目标值小于0，表示在该位置已经打过了
        if target_value < 0:
            print(
                f"YOU ALREADY PUT A HOLE IN SHIP NUMBER {abs(target_value)} AT THAT POINT."
            )

        # 如果目标值小于等于0，表示没有击中船只
        if target_value <= 0:
            print("SPLASH! TRY AGAIN.")
            splashes += 1
            continue

        # 如果击中船只，打印击中信息，并增加击中次数
        print(f"A DIRECT HIT ON SHIP NUMBER {target_value}")
        hits += 1
        # 在目标位置设置值为目标值的相反数，表示击中
        set_value_at(-target_value, target, sea)

        # 如果击沉了船只，打印相关信息
        if not has_ship(sea, target_value):
            print("AND YOU SUNK IT. HURRAH FOR THE GOOD GUYS.")
            print("SO FAR, THE BAD GUYS HAVE LOST")
            print(
                f"{count_sunk(sea, 1, 2)} DESTROYER(S),",
                f"{count_sunk(sea, 3, 4)} CRUISER(S),",
                f"AND {count_sunk(sea, 5, 6)} AIRCRAFT CARRIER(S).",
            )

        # 如果海域中还有船只，打印当前的击中/未击中比例
        if any(has_ship(sea, code) for code in range(1, 7)):
            print(f"YOUR CURRENT SPLASH/HIT RATIO IS {splashes}/{hits}")
            continue

        # 如果海域中没有船只了，打印最终的击中/未击中比例
        print(
            "YOU HAVE TOTALLY WIPED OUT THE BAD GUYS' FLEET "
            f"WITH A FINAL SPLASH/HIT RATIO OF {splashes}/{hits}"
        )

        # 如果没有未击中的次数，打印祝贺信息
        if not splashes:
            print("CONGRATULATIONS -- A DIRECT HIT EVERY TIME.")

        # 打印分隔线并结束循环
        print("\n****************************")
        break
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```