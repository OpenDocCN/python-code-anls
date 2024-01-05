# `d:/src/tocomm/basic-computer-games\09_Battle\python\battle.py`

```
        continue  # 继续循环直到找到非零向量
        return vector  # 返回非零向量
        # 继续循环，直到找到合适的位置
        continue

    # 返回向量
    return vector


def add_vector(point: PointType, vector: VectorType) -> PointType:
    # 返回点加上向量后的新点
    return (point[0] + vector[0], point[1] + vector[1])


def place_ship(sea: SeaType, size: int, code: int) -> None:
    # 循环直到找到合适的位置
    while True:
        # 随机生成起始点
        start = (randrange(1, SEA_WIDTH + 1), randrange(1, SEA_WIDTH + 1))
        # 随机生成方向向量
        vector = random_vector()

        # 获取潜在的船的点
        point = start
        points = []

        # 循环生成船的各个点
        for _ in range(size):
            # 将点加上向量得到新的点
            point = add_vector(point, vector)
        points.append(point)  # 将点添加到列表中

        if not all([is_within_sea(point, sea) for point in points]) or any(
            [value_at(point, sea) for point in points]
        ):
            # 如果点超出海域范围或者与其他船重叠，则继续循环
            continue

        # 找到有效位置，现在实际放置船只
        for point in points:
            set_value_at(code, point, sea)  # 在指定位置设置值

        break  # 结束循环


def print_encoded_sea(sea: SeaType) -> None:
    for x in range(len(sea)):
        # 打印编码后的海域
        print(" ".join([str(sea[y][x]) for y in range(len(sea) - 1, -1, -1)]))
# 检查给定的点是否在海域范围内，返回布尔值
def is_within_sea(point: PointType, sea: SeaType) -> bool:
    return (1 <= point[0] <= len(sea)) and (1 <= point[1] <= len(sea)

# 检查海域中是否存在指定的船只代码，返回布尔值
def has_ship(sea: SeaType, code: int) -> bool:
    return any(code in row for row in sea)

# 统计海域中沉没的船只数量，返回整数
def count_sunk(sea: SeaType, *codes: int) -> int:
    return sum(not has_ship(sea, code) for code in codes)

# 获取给定点在海域中的值，返回整数
def value_at(point: PointType, sea: SeaType) -> int:
    return sea[point[1] - 1][point[0] - 1]

# 设置给定点在海域中的值
def set_value_at(value: int, point: PointType, sea: SeaType) -> None:
    sea[point[1] - 1][point[0] - 1] = value
def get_next_target(sea: SeaType) -> PointType:
    # 无限循环，直到满足条件才会退出循环
    while True:
        try:
            # 从用户输入中获取猜测的坐标
            guess = input("? ")
            # 将用户输入的坐标字符串以逗号分隔，得到坐标字符串列表
            point_str_list = guess.split(",")

            # 如果坐标字符串列表长度不为2，抛出值错误异常
            if len(point_str_list) != 2:
                raise ValueError()

            # 将坐标字符串列表中的两个字符串转换为整数，组成坐标点
            point = (int(point_str_list[0]), int(point_str_list[1]))

            # 如果坐标点不在海域范围内，抛出值错误异常
            if not is_within_sea(point, sea):
                raise ValueError()

            # 如果坐标点满足条件，返回该坐标点
            return point
        except ValueError:
            # 捕获值错误异常，打印错误信息
            print(
                f"INVALID. SPECIFY TWO NUMBERS FROM 1 TO {len(sea)}, SEPARATED BY A COMMA."
            )
def setup_ships(sea: SeaType) -> None:
    # 在海域中放置一艘长度为 DESTROYER_LENGTH 的驱逐舰
    place_ship(sea, DESTROYER_LENGTH, 1)
    # 在海域中放置一艘长度为 DESTROYER_LENGTH 的驱逐舰
    place_ship(sea, DESTROYER_LENGTH, 2)
    # 在海域中放置一艘长度为 CRUISER_LENGTH 的巡洋舰
    place_ship(sea, CRUISER_LENGTH, 3)
    # 在海域中放置一艘长度为 CRUISER_LENGTH 的巡洋舰
    place_ship(sea, CRUISER_LENGTH, 4)
    # 在海域中放置一艘长度为 AIRCRAFT_CARRIER_LENGTH 的航空母舰
    place_ship(sea, AIRCRAFT_CARRIER_LENGTH, 5)
    # 在海域中放置一艘长度为 AIRCRAFT_CARRIER_LENGTH 的航空母舰
    place_ship(sea, AIRCRAFT_CARRIER_LENGTH, 6)


def main() -> None:
    # 创建一个二维元组表示的海域，元组中每个元素都是长度为 SEA_WIDTH 的列表
    sea = tuple([0 for _ in range(SEA_WIDTH)] for _ in range(SEA_WIDTH))
    # 在海域中放置敌方舰队
    setup_ships(sea)
    # 打印游戏开始信息
    print(
        """
                BATTLE
CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY

THE FOLLOWING CODE OF THE BAD GUYS' FLEET DISPOSITION
HAS BEEN CAPTURED BUT NOT DECODED:
    )
    # 初始化变量 splashes 和 hits
    splashes = 0
    hits = 0

    # 进入游戏循环
    while True:
        # 获取下一个目标位置
        target = get_next_target(sea)
        # 获取目标位置的值
        target_value = value_at(target, sea)

        # 如果目标位置的值小于 0，表示没有命中
        if target_value < 0:
            print(
                f"YOU ALREADY PUT A HOLE IN SHIP NUMBER {abs(target_value)} AT THAT POINT."
            )  # 打印已经在目标位置上打了一个洞的消息

        if target_value <= 0:  # 如果目标值小于等于0
            print("SPLASH! TRY AGAIN.")  # 打印未击中目标的消息
            splashes += 1  # 增加未击中目标的次数
            continue  # 继续下一次循环

        print(f"A DIRECT HIT ON SHIP NUMBER {target_value}")  # 打印直接击中目标的消息
        hits += 1  # 增加击中目标的次数
        set_value_at(-target_value, target, sea)  # 在目标位置上设置值为负目标值，表示击中目标

        if not has_ship(sea, target_value):  # 如果目标位置上没有船只
            print("AND YOU SUNK IT. HURRAH FOR THE GOOD GUYS.")  # 打印击沉目标的消息
            print("SO FAR, THE BAD GUYS HAVE LOST")  # 打印敌方已经失去的船只数量
            print(
                f"{count_sunk(sea, 1, 2)} DESTROYER(S),",
                f"{count_sunk(sea, 3, 4)} CRUISER(S),",
                f"AND {count_sunk(sea, 5, 6)} AIRCRAFT CARRIER(S).",
            )  # 打印不同类型船只的击沉数量
        )

        # 如果海域中有任何一个位置有船只，则打印当前的击中/溅起比率
        if any(has_ship(sea, code) for code in range(1, 7)):
            print(f"YOUR CURRENT SPLASH/HIT RATIO IS {splashes}/{hits}")
            continue

        # 打印最终的溅起/击中比率，表示成功摧毁了敌方舰队
        print(
            "YOU HAVE TOTALLY WIPED OUT THE BAD GUYS' FLEET "
            f"WITH A FINAL SPLASH/HIT RATIO OF {splashes}/{hits}"
        )

        # 如果没有溅起，则打印直接命中每次的祝贺语
        if not splashes:
            print("CONGRATULATIONS -- A DIRECT HIT EVERY TIME.")

        # 打印分隔线并结束游戏
        print("\n****************************")
        break


if __name__ == "__main__":
    main()
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```