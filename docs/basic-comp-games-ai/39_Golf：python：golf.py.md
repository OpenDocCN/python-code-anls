# `39_Golf\python\golf.py`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 以二进制模式打开文件，读取文件内容，封装成字节流
    使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面的内容创建 ZIP 对象，以只读模式打开
    遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象中的文件名列表，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回结果字典
# 大量的随机化、"业务规则"和运气影响游戏的进行。概率在代码中有注释。

注意：'courseInfo'、'clubs'和'scoreCard'数组中都包括一个空对象，以便从1开始索引。像所有优秀的程序员一样，我们从零开始计数，但在这种情境中，当第一洞的编号为1时，从1开始更自然。

    |-----------------------------|
    |            沙坑             |
    |   ----------------------    |
    |   |                     |   |
    | r |        =  =         | r |
    | o |     =        =      | o |
    | u |    =    .     =     | u |
    | g |    =   绿色   =     | g |
    | h |     =        =      | h |
    |   |        =  =         |   |
    |   |                     |   |
    |   |                     |   |
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从给定的文件名读取二进制数据，并将其封装成字节流对象
    使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里的内容创建一个 ZIP 对象，以只读模式打开
    遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象中的文件名列表，读取每个文件的数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回包含文件名到数据的字典
# 计算球的新位置，给定球的位置、击球距离和偏离直线的角度（hook或slice）。
# 对于右撇子：
# Slice：正角度=球向右移动
# Hook：负角度=球向左移动

# 杯子始终在点：0,0。
# 我们使用 atan2 来计算杯子和球之间的角度。
# 将杯子的向量设置为0，-1在360度圆上等同于：
# 0度=12点钟；90度=3点钟；180度=6点钟；270度=9点钟
# 杯子和球之间的反向角度是PI的差值（使用弧度）。

# 给定角度和击球距离（斜边），我们使用余弦来计算三角形的对边和邻边，即球的新位置。

#         0
#         |
# 270 - 杯子 - 90
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从给定的文件名读取二进制数据，并封装成字节流
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以便后续操作
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象，释放资源
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
# 导入 enum 模块，用于创建枚举类型
import enum
# 导入 math 模块，用于数学运算
import math
# 导入 random 模块，用于生成随机数
import random
import time  # 导入时间模块
from dataclasses import dataclass  # 导入 dataclass 模块
from functools import partial  # 导入 partial 函数模块
from typing import Any, Callable, List, NamedTuple, Tuple  # 导入类型提示模块


def clear_console() -> None:  # 定义清空控制台的函数，不返回任何值
    print("\033[H\033[J", end="")  # 打印特殊字符来清空控制台


class Point(NamedTuple):  # 定义一个名为 Point 的命名元组
    x: int  # 元组中的 x 属性为整数类型
    y: int  # 元组中的 y 属性为整数类型


class GameObjType(enum.Enum):  # 定义一个名为 GameObjType 的枚举类
    BALL = enum.auto()  # 枚举类中的 BALL 属性
    CUP = enum.auto()  # 枚举类中的 CUP 属性
    GREEN = enum.auto()  # 枚举类中的 GREEN 属性
    FAIRWAY = enum.auto()  # 枚举类中的 FAIRWAY 属性
    # 定义枚举类型，表示游戏对象的类型
    ROUGH = enum.auto()  # 表示粗糙地形
    TREES = enum.auto()  # 表示树木
    WATER = enum.auto()  # 表示水域
    SAND = enum.auto()   # 表示沙地

class CircleGameObj(NamedTuple):
    # 定义圆形游戏对象的属性
    X: int  # 圆心横坐标
    Y: int  # 圆心纵坐标
    Radius: int  # 半径
    Type: GameObjType  # 类型，引用枚举类型

class RectGameObj(NamedTuple):
    # 定义矩形游戏对象的属性
    X: int  # 矩形左上角横坐标
    Y: int  # 矩形左上角纵坐标
    Width: int  # 矩形宽度
    Length: int  # 矩形长度
    Type: GameObjType  # 定义了一个类型注解，指定了变量的类型为GameObjType

Ball = CircleGameObj  # 将CircleGameObj赋值给Ball变量
Hazard = CircleGameObj  # 将CircleGameObj赋值给Hazard变量

class HoleInfo(NamedTuple):  # 定义了一个名为HoleInfo的类，继承自NamedTuple
    hole: int  # 定义了一个名为hole的属性，类型为int
    yards: int  # 定义了一个名为yards的属性，类型为int
    par: int  # 定义了一个名为par的属性，类型为int
    hazards: List[Hazard]  # 定义了一个名为hazards的属性，类型为List，其中元素类型为Hazard
    description: str  # 定义了一个名为description的属性，类型为str

class HoleGeometry(NamedTuple):  # 定义了一个名为HoleGeometry的类，继承自NamedTuple
    cup: CircleGameObj  # 定义了一个名为cup的属性，类型为CircleGameObj
    green: CircleGameObj  # 定义了一个名为green的属性，类型为CircleGameObj
    fairway: RectGameObj  # 定义了一个名为fairway的属性，类型为RectGameObj
    rough: RectGameObj  # 定义了一个名为rough的属性，类型为RectGameObj
    hazards: List[Hazard]  # 定义了一个名为hazards的列表，其中包含Hazard类型的元素


@dataclass  # 使用dataclass装饰器定义了一个名为Plot的数据类
class Plot:
    x: int  # 定义了一个名为x的整数类型属性
    y: int  # 定义了一个名为y的整数类型属性
    offline: int  # 定义了一个名为offline的整数类型属性


def get_distance(pt1: Point, pt2: Point) -> float:
    """distance between 2 points"""
    # 计算两点之间的距离
    return math.sqrt(math.pow((pt2.x - pt1.x), 2) + math.pow((pt2.y - pt1.y), 2))


def is_in_rectangle(pt: CircleGameObj, rect: RectGameObj) -> bool:
    # only true if its completely inside
    # 判断圆形游戏对象是否完全位于矩形游戏对象内部
    return (
        (pt.X > rect.X)
        and (pt.X < rect.X + rect.Width)  # 判断圆形游戏对象的X坐标是否在矩形游戏对象的X坐标范围内
        and (pt.Y > rect.Y)  # 检查点的 Y 坐标是否大于矩形的 Y 坐标
        and (pt.Y < rect.Y + rect.Length)  # 检查点的 Y 坐标是否小于矩形的 Y 坐标加上矩形的长度
    )


def to_radians(angle: float) -> float:
    return angle * (math.pi / 180.0)  # 将角度转换为弧度


def to_degrees_360(angle: float) -> float:
    """radians to 360 degrees"""
    deg = angle * (180.0 / math.pi)  # 将弧度转换为角度
    if deg < 0.0:
        deg += 360.0  # 如果角度小于0，则加上360度
    return deg


def odds(x: int) -> bool:
    # chance an integer is <= the given argument
    # between 1-100  # 返回一个整数是否小于等于给定参数的概率，范围在1-100之间
    return random.randint(1, 101) <= x  # 返回一个随机数，判断是否小于等于给定的数x

# THE COURSE
CourseInfo = [
    HoleInfo(0, 0, 0, [], ""),  # 创建一个空的HoleInfo对象，使索引1等于第1个洞
    # -------------------------------------------------------- front 9
    HoleInfo(
        1,
        361,
        4,
        [
            Hazard(20, 100, 10, GameObjType.TREES),  # 创建一个Hazard对象，表示树木障碍物
            Hazard(-20, 80, 10, GameObjType.TREES),  # 创建一个Hazard对象，表示树木障碍物
            Hazard(-20, 100, 10, GameObjType.TREES),  # 创建一个Hazard对象，表示树木障碍物
        ],
        "There are a couple of trees on the left and right.",  # 描述该洞的信息
    ),
    HoleInfo(
        2,
        389,  # 第3个洞的编号
        4,    # 第3个洞的标准杆数
        [Hazard(0, 160, 20, GameObjType.WATER)],  # 第3个洞的障碍物信息，包括位置和类型
        "There is a large water hazard across the fairway about 150 yards.",  # 第3个洞的描述信息
    ),
    HoleInfo(
        3,    # 第4个洞的编号
        206,  # 第4个洞的长度
        3,    # 第4个洞的标准杆数
        [   # 第4个洞的障碍物信息列表
            Hazard(20, 20, 5, GameObjType.WATER),  # 水障碍物的位置和类型
            Hazard(-20, 160, 10, GameObjType.WATER),  # 水障碍物的位置和类型
            Hazard(10, 12, 5, GameObjType.SAND),  # 沙坑障碍物的位置和类型
        ],
        "There is some sand and water near the green.",  # 第4个洞的描述信息
    ),
    HoleInfo(
        4,    # 第5个洞的编号
        500,  # 第5个洞的长度
        5,    # 第5个洞的标准杆数
        [Hazard(-14, 12, 12, GameObjType.SAND)],  # 创建一个包含 Hazard 对象的列表，表示球道上的危险区域
        "There's a bunker to the left of the green.",  # 提示信息，描述球道上的情况
    ),
    HoleInfo(
        5,  # 球道编号
        408,  # 球道长度
        4,  # 球道标准杆数
        [
            Hazard(20, 120, 20, GameObjType.TREES),  # 创建一个 Hazard 对象，表示球道上的树木区域
            Hazard(20, 160, 20, GameObjType.TREES),  # 创建另一个 Hazard 对象，表示球道上的树木区域
            Hazard(10, 20, 5, GameObjType.SAND),  # 创建一个 Hazard 对象，表示球道上的沙坑区域
        ],
        "There are some trees to your right.",  # 提示信息，描述球道上的情况
    ),
    HoleInfo(
        6,  # 球道编号
        359,  # 球道长度
        4,  # 球道标准杆数
        [Hazard(14, 0, 4, GameObjType.SAND), Hazard(-14, 0, 4, GameObjType.SAND)],  # 创建两个 Hazard 对象，表示球道上的沙坑区域
        "",  # 空的提示信息
    ),
    # 创建一个名为HoleInfo的对象，包含编号、长度、难度等信息，以及沙坑的位置和类型
    HoleInfo(
        7,
        424,
        5,
        [
            Hazard(20, 200, 10, GameObjType.SAND),  # 创建一个名为Hazard的对象，表示沙坑的位置和类型
            Hazard(10, 180, 10, GameObjType.SAND),  # 创建一个名为Hazard的对象，表示沙坑的位置和类型
            Hazard(20, 160, 10, GameObjType.SAND),  # 创建一个名为Hazard的对象，表示沙坑的位置和类型
        ],
        "There are several sand traps along your right.",  # 描述沙坑位置的字符串
    ),
    # 创建一个名为HoleInfo的对象，包含编号、长度、难度等信息，以及树木障碍的位置和类型
    HoleInfo(8, 388, 4, [Hazard(-20, 340, 10, GameObjType.TREES)], ""),  # 创建一个名为Hazard的对象，表示树木障碍的位置和类型
    # 创建一个名为HoleInfo的对象，包含编号、长度、难度等信息，以及树木障碍和沙坑的位置和类型
    HoleInfo(
        9,
        196,
        3,
        [Hazard(-30, 180, 20, GameObjType.TREES), Hazard(14, -8, 5, GameObjType.SAND)],  # 创建名为Hazard的对象，表示树木障碍和沙坑的位置和类型
        "",  # 描述为空字符串
    ),
    # -------------------------------------------------------- back 9
    # 创建 HoleInfo 对象，表示第10洞的信息
    HoleInfo(
        hole=10,  # 洞号
        yards=400,  # 码数
        par=4,  # 标准杆数
        hazards=[  # 障碍物列表
            Hazard(-14, -8, 5, GameObjType.SAND),  # 沙坑障碍物
            Hazard(14, -8, 5, GameObjType.SAND),  # 沙坑障碍物
        ],
        description="",  # 描述
    ),
    # 创建 HoleInfo 对象，表示第11洞的信息
    HoleInfo(
        11,  # 洞号
        560,  # 码数
        5,  # 标准杆数
        [  # 障碍物列表
            Hazard(-20, 400, 10, GameObjType.TREES),  # 树木障碍物
            Hazard(-10, 380, 10, GameObjType.TREES),  # 树木障碍物
            Hazard(-20, 260, 10, GameObjType.TREES),  # 树木障碍物
            Hazard(-20, 200, 10, GameObjType.TREES),  # 树木障碍物
            Hazard(-10, 180, 10, GameObjType.TREES),  # 在位置(-10, 180)处添加一个类型为树木的障碍物
            Hazard(-20, 160, 10, GameObjType.TREES),  # 在位置(-20, 160)处添加一个类型为树木的障碍物
        ],
        "Lots of trees along the left of the fairway.",  # 在球道左侧有很多树木。

    HoleInfo(
        12,  # 第12洞
        132,  # 长度132码
        3,   # 杆数为3
        [
            Hazard(-10, 120, 10, GameObjType.WATER),  # 在位置(-10, 120)处添加一个类型为水的障碍物
            Hazard(-5, 100, 10, GameObjType.SAND),    # 在位置(-5, 100)处添加一个类型为沙坑的障碍物
        ],
        "There is water and sand directly in front of you. A good drive should clear both.",  # 在你正前方有水和沙坑。一次好的开球应该能清除两者。

    HoleInfo(
        13,  # 第13洞
        357,  # 长度357码
        4,    # 杆数为4
        [
            Hazard(-20, 200, 10, GameObjType.TREES),  # 在特定位置创建树木障碍物
            Hazard(-10, 180, 10, GameObjType.TREES),  # 在特定位置创建树木障碍物
            Hazard(-20, 160, 10, GameObjType.TREES),  # 在特定位置创建树木障碍物
            Hazard(14, 12, 8, GameObjType.SAND),  # 在特定位置创建沙坑障碍物
        ],
        "",  # 空字符串
    ),
    HoleInfo(14, 294, 4, [Hazard(0, 20, 10, GameObjType.SAND)], ""),  # 在特定位置创建沙坑障碍物
    HoleInfo(
        15,
        475,
        5,
        [Hazard(-20, 20, 10, GameObjType.WATER), Hazard(10, 20, 10, GameObjType.SAND)],  # 在特定位置创建水障碍物和沙坑障碍物
        "Some sand and water near the green.",  # 描述信息
    ),
    HoleInfo(16, 375, 4, [Hazard(-14, -8, 5, GameObjType.SAND)], ""),  # 在特定位置创建沙坑障碍物
    HoleInfo(
        17,
        180,
        3,
        [
            Hazard(20, 100, 10, GameObjType.TREES),  # 创建一个名为Hazard的对象，表示树木障碍物
            Hazard(-20, 80, 10, GameObjType.TREES),  # 创建另一个名为Hazard的对象，表示树木障碍物
        ],
        "",  # 空字符串
    ),
    HoleInfo(
        18,
        550,
        5,
        [Hazard(20, 30, 15, GameObjType.WATER)],  # 创建一个名为Hazard的对象，表示水障碍物
        "There is a water hazard near the green.",  # 描述水障碍物在果岭附近
    ),
]


# -------------------------------------------------------- bitwise Flags
dub = 0b00000000000001  # 用二进制表示的标志位，表示某种状态或属性
hook = 0b00000000000010  # 用二进制表示的标志位，表示某种状态或属性
slice_ = 0b00000000000100  # 用二进制表示的标志位，表示某种状态或属性
# 通过二进制表示球的状态，表示球是否通过杯口
passed_cup = 0b00000000001000
# 通过二进制表示球的状态，表示球是否在杯中
in_cup = 0b00000000010000
# 通过二进制表示球的状态，表示球是否在球道上
on_fairway = 0b00000000100000
# 通过二进制表示球的状态，表示球是否在果岭上
on_green = 0b00000001000000
# 通过二进制表示球的状态，表示球是否在球道外
in_rough = 0b00000010000000
# 通过二进制表示球的状态，表示球是否在沙坑中
in_sand = 0b00000100000000
# 通过二进制表示球的状态，表示球是否在树丛中
in_trees = 0b00001000000000
# 通过二进制表示球的状态，表示球是否在水中
in_water = 0b00010000000000
# 通过二进制表示球的状态，表示球是否出界
out_of_bounds = 0b00100000000000
# 通过二进制表示球的状态，表示球是否有幸运
luck = 0b01000000000000
# 通过二进制表示球的状态，表示球是否完成一杆进洞
ace = 0b10000000000000

# 创建 Golf 类
class Golf:
    # 球：Ball 类型
    ball: Ball
    # 洞号：整数类型，默认为 0
    hole_num: int = 0
    # 击球数：整数类型，默认为 0
    stroke_num: int = 0
    # 障碍：整数类型，默认为 0
    handicap: int = 0
    # 球员难度：整数类型，默认为 0
    player_difficulty: int = 0
    # 洞的几何形状：HoleGeometry 类型
    hole_geometry: HoleGeometry
    # 所有球道宽度为40码，延伸至杯子外5码，并且周围有5码的草地
    fairway_width: int = 40
    fairway_extension: int = 5
    rough_amt: int = 5

    # ScoreCard记录每一杆击球后球的位置
    # 每个洞都有一个新的列表
    # 包括一个空列表，使得索引1 == 第1个洞
    score_card: List[List[Ball]] = [[]]

    # 你的球袋
    clubs: List[Tuple[str, int]] = [
        ("", 0),
        # 名称，平均码数
        ("Driver", 250),
        ("3 Wood", 225),
        ("5 Wood", 200),
        ("Hybrid", 190),
        ("4 Iron", 170),  # 创建一个元组，包含高尔夫球杆名称和对应的距离
        ("7 Iron", 150),  # 创建一个元组，包含高尔夫球杆名称和对应的距离
        ("9 Iron", 125),  # 创建一个元组，包含高尔夫球杆名称和对应的距离
        ("Pitching wedge", 110),  # 创建一个元组，包含高尔夫球杆名称和对应的距离
        ("Sand wedge", 75),  # 创建一个元组，包含高尔夫球杆名称和对应的距离
        ("Putter", 10),  # 创建一个元组，包含高尔夫球杆名称和对应的距离
    ]

    def __init__(self) -> None:  # 类的初始化方法
        print(" ")  # 打印空行
        print('          8""""8 8"""88 8     8"""" ')  # 打印文本
        print('          8    " 8    8 8     8     ')  # 打印文本
        print("          8e     8    8 8e    8eeee ")  # 打印文本
        print("          88  ee 8    8 88    88    ")  # 打印文本
        print("          88   8 8    8 88    88    ")  # 打印文本
        print("          88eee8 8eeee8 88eee 88    ")  # 打印文本
        print(" ")  # 打印空行
        print("Welcome to the Creative Computing Country Club,")  # 打印欢迎词
        print("an eighteen hole championship layout located a short")  # 打印文本
        print("distance from scenic downtown Lambertville, New Jersey.")  # 打印文本
        # 打印游戏说明
        print("The game will be explained as you play.")
        # 打印祝福语
        print("Enjoy your game! See you at the 19th hole...")
        # 打印空行
        print(" ")
        # 提示玩家可以随时输入QUIT退出游戏
        print("Type QUIT at any time to leave the game.")
        # 提示玩家可以随时输入BAG查看球袋中的球杆
        print("Type BAG at any time to review the clubs in your bag.")
        # 打印空行
        print(" ")

        # 等待玩家按下任意键继续游戏
        input("Press any key to continue.")
        # 清空控制台
        clear_console()
        # 调用start_game方法开始游戏

    def start_game(self) -> None:
        # 打印空行
        print(" ")
        # 打印球袋标题
        print("              YOUR BAG")
        # 调用review_bag方法查看球袋中的球杆
        self.review_bag()
        # 提示玩家可以随时输入BAG查看球袋中的球杆
        print("Type BAG at any time to review the clubs in your bag.")
        # 打印空行
        print(" ")

        # 等待玩家按下任意键继续游戏
        input("Press any key to continue.")
        # 清空控制台
        clear_console()
        self.ask_handicap()  # 调用 ask_handicap 方法，询问用户的高尔夫球差点

    def ask_handicap(self) -> None:  # 定义 ask_handicap 方法，用于询问用户的高尔夫球差点
        print(" ")  # 打印空行

        self.ask(  # 调用 ask 方法，询问用户的高尔夫球差点
            "PGA handicaps range from 0 to 30.\nWhat is your handicap?",  # 提示用户输入高尔夫球差点的范围
            0,  # 最小值为 0
            30,  # 最大值为 30
            self.set_handicap_ask_difficulty,  # 用户输入后调用 set_handicap_ask_difficulty 方法
        )

    def set_handicap_ask_difficulty(self, i: int) -> None:  # 定义 set_handicap_ask_difficulty 方法，用于设置用户的高尔夫球差点并询问困难程度
        self.handicap = i  # 将用户输入的高尔夫球差点赋值给 self.handicap
        print(" ")  # 打印空行

        self.ask(  # 调用 ask 方法，询问用户的高尔夫困难程度
            (
                "Common difficulties at golf include:\n"  # 提示用户选择高尔夫的常见困难
                "1=Hook, 2=Slice, 3=Poor Distance, 4=Trap Shots, 5=Putting\n"
        "Which one is your worst?"
    ),  # 结束对话框的文本内容
    1,  # 对话框的类型，1表示是一个问题对话框
    5,  # 对话框的按钮类型，5表示是一个Yes/No对话框
    self.set_difficulty_and_hole,  # 当用户点击Yes按钮时，调用set_difficulty_and_hole方法
)

def set_difficulty_and_hole(self, j: int) -> None:
    self.player_difficulty = j  # 将用户选择的难度赋值给player_difficulty变量
    clear_console()  # 清空控制台
    self.new_hole()  # 调用new_hole方法开始新的一轮比赛

def new_hole(self) -> None:
    self.hole_num += 1  # 当前洞数加1
    self.stroke_num = 0  # 击球数重置为0

    info: HoleInfo = CourseInfo[self.hole_num]  # 从CourseInfo中获取当前洞的信息

    yards: int = info.yards  # 获取当前洞的码数
    # from tee to cup  # 从发球区到洞口
        # 创建一个圆形游戏对象，代表杯子
        cup = CircleGameObj(0, 0, 0, GameObjType.CUP)
        # 创建一个圆形游戏对象，代表果岭上的草坪
        green = CircleGameObj(0, 0, 10, GameObjType.GREEN)

        # 创建一个矩形游戏对象，代表球道
        fairway = RectGameObj(
            0 - int(self.fairway_width / 2),
            0 - (green.Radius + self.fairway_extension),
            self.fairway_width,
            yards + (green.Radius + self.fairway_extension) + 1,
            GameObjType.FAIRWAY,
        )

        # 创建一个矩形游戏对象，代表球道外的粗糙地形
        rough = RectGameObj(
            fairway.X - self.rough_amt,
            fairway.Y - self.rough_amt,
            fairway.Width + (2 * self.rough_amt),
            fairway.Length + (2 * self.rough_amt),
            GameObjType.ROUGH,
        )

        # 创建一个球形游戏对象，代表高尔夫球
        self.ball = Ball(0, yards, 0, GameObjType.BALL)
        self.score_card_start_new_hole()  # 调用score_card_start_new_hole方法，开始新的一轮比赛

        self.hole_geometry = HoleGeometry(cup, green, fairway, rough, info.hazards)  # 创建HoleGeometry对象，用于描述球洞的几何特征

        print(f"                |> {self.hole_num}")  # 打印当前球洞号码
        print("                |        ")  # 打印空行
        print("                |        ")  # 打印空行
        print("          ^^^^^^^^^^^^^^^")  # 打印分隔线

        print(
            f"Hole #{self.hole_num}. You are at the tee. Distance {info.yards} yards, par {info.par}."
        )  # 打印当前球洞信息，包括球洞号码、距离和标准杆数
        print(info.description)  # 打印球洞描述信息

        self.tee_up()  # 调用tee_up方法，准备击球

    def set_putter_and_stroke(self, strength: float) -> None:  # 设置推杆和击球力度的方法
        putter = self.clubs[self.putt]  # 选择推杆
        self.stroke((putter[1] * (strength / 10.0)), self.putt)  # 根据力度和推杆长度计算击球力度，并进行击球
    # 定义一个方法，用于询问用户选择哪个球杆
    def ask_gauge(self, c: int) -> None:
        # 将选择的球杆赋值给self.club
        self.club = self.clubs[c]

        # 打印选定球杆的平均击球距离
        print(" ")
        print(f"[{self.club[0].upper()}: average {self.club[1]} yards]")

        # 创建一个部分函数，用于制造击球动作
        foo = partial(self.make_stroke, c=c)

        # 询问用户按全力击球的百分比来测量距离
        self.ask(
            "Now gauge your distance by a percentage of a full swing. (1-10)",
            1,
            10,
            foo,
        )

    # 定义一个方法，用于进行击球动作
    def make_stroke(self, strength: float, c: int) -> None:
        # 根据用户输入的力量和选定球杆的平均击球距离进行击球
        self.stroke((self.club[1] * (strength / 10.0)), c)

    # 定义一个方法，用于准备击球
    def tee_up(self) -> None:
        # 如果球在果岭上并且不在沙坑中，自动选择推杆
        # 否则询问球杆和挥杆力度
        if self.is_on_green(self.ball) and not self.is_in_hazard(
            self.ball, GameObjType.SAND
        ):
            self.putt = 10
            print("[推杆：平均10码]")
            if odds(20):
                msg = "保持低头。\n"
            else:
                msg = ""

            self.ask(
                msg + "选择推杆力度。 (1-10)",
                1,
                10,
                self.set_putter_and_stroke,
            )
        else:
            self.ask("你选择什么球杆？ (1-10)", 1, 10, self.ask_gauge)
                or self.is_in_hazard(self.ball, GameObjType.WATER)
            )
            and club_index not in [7, 8, 9]
        ):
            flags |= wrong_club

        # update the club index
        self.club_index = club_index

        # update the club amount
        self.club_amt = club_amt

        # update the flags
        self.flags = flags
            )
            and not (club_index == 8 or club_index == 9)  # 如果俱乐部索引不是8或9
            and odds(40)  # 以40%的概率
        ):
            flags |= dub  # 设置标志位dub

        # trap difficulty
        if (
            self.is_in_hazard(self.ball, GameObjType.SAND)  # 如果球在沙坑中
            and self.player_difficulty == 4  # 并且玩家难度为4
        ) and odds(20):  # 以20%的概率
            flags |= dub  # 设置标志位dub

        # hook/slice
        # There's 10% chance of a hook or slice
        # if it's a known player_difficulty then increase chance to 30%
        # if it's a putt & putting is a player_difficulty increase to 30%

        rand_hook_slice: bool  # 定义一个布尔型变量rand_hook_slice
        if (
            # 检查玩家难度是否为1或2，或者难度为5且球在绿色区域上
            if self.player_difficulty == 1
            or self.player_difficulty == 2
            or (self.player_difficulty == 5 and self.is_on_green(self.ball))
        ):
            # 根据概率生成随机的hook或slice值
            rand_hook_slice = odds(30)
        else:
            # 根据概率生成随机的hook或slice值
            rand_hook_slice = odds(10)

        # 根据生成的随机值进行下一步操作
        if rand_hook_slice:
            # 根据玩家难度和概率生成hook或slice值，并更新flags
            if self.player_difficulty == 1:
                if odds(80):
                    flags |= hook
                else:
                    flags |= slice_
            elif self.player_difficulty == 2:
                if odds(80):
                    flags |= slice_
                else:
                    flags |= hook
            else:
                # 其他情况下不做任何操作
                if odds(50):  # 如果随机数为50%的概率
                    flags |= hook  # 将hook标志位加入到flags中
                else:
                    flags |= slice_  # 否则将slice_标志位加入到flags中

        # beginner's luck !  # 初学者的幸运！
        # 如果手动让球数大于15，有10%的几率避免所有错误
        if (self.handicap > 15) and (odds(10)):
            flags |= luck  # 将luck标志位加入到flags中

        # ace  # 王牌
        # 在标准杆为3的球洞上，有10%的几率打出王牌
        if CourseInfo[self.hole_num].par == 3 and odds(10) and self.stroke_num == 1:
            flags |= ace  # 将ace标志位加入到flags中

        # distance:  # 距离：
        # 如果手动让球数小于15，有50%的几率达到球杆平均距离，
        # 25%的几率超过平均距离，25%的几率落后于平均距离
        # 如果手动让球数大于15，有25%的几率达到球杆平均距离，
        # 75%的几率落后于平均距离
        # The greater the handicap, the more the ball falls short
        # If poor distance is a known player_difficulty, then reduce distance by 10%

        distance: float  # 声明一个浮点型变量 distance
        rnd = random.randint(1, 101)  # 生成一个1到101之间的随机整数赋值给变量 rnd

        if self.handicap < 15:  # 如果 self.handicap 小于 15
            if rnd <= 25:  # 如果 rnd 小于等于 25
                distance = club_amt - (club_amt * (self.handicap / 100.0))  # 计算 distance
            elif rnd > 25 and rnd <= 75:  # 如果 rnd 大于 25 且小于等于 75
                distance = club_amt  # distance 等于 club_amt
            else:  # 其他情况
                distance = club_amt + (club_amt * 0.10)  # 计算 distance
        else:  # 如果 self.handicap 不小于 15
            if rnd <= 75:  # 如果 rnd 小于等于 75
                distance = club_amt - (club_amt * (self.handicap / 100.0))  # 计算 distance
            else:  # 其他情况
                distance = club_amt  # distance 等于 club_amt

        if self.player_difficulty == 3 and odds(80):  # 如果 self.player_difficulty 等于 3 并且 odds(80) 返回 True
            distance = distance * 0.80  # 将距离乘以0.80，以模拟球的飞行距离

        if (flags & luck) == luck:  # 如果标志中包含幸运标志
            distance = club_amt  # 将距离设置为俱乐部的数量

        # angle
        # 对于所有击球，可能会有4度的“漂移”
        # 钩或切增加5-10度之间的角度，
        # 钩使用负度数
        angle = random.randint(0, 5)  # 生成0到5之间的随机角度
        if (flags & slice_) == slice_:  # 如果标志中包含切球标志
            angle = random.randint(5, 11)  # 生成5到11之间的随机角度
        if (flags & hook) == hook:  # 如果标志中包含钩标志
            angle = 0 - random.randint(5, 11)  # 生成-5到-11之间的随机角度
        if (flags & luck) == luck:  # 如果标志中包含幸运标志
            angle = 0  # 将角度设置为0

        plot = self.plot_ball(self.ball, distance, angle)  # 根据距离和角度绘制球的位置
        # 计算新位置
        if (flags & luck) == luck and plot.y > 0:  # 如果标志中包含幸运标志并且绘制的y坐标大于0
            plot.y = 2  # 设置变量 plot 的 y 值为 2

        flags = self.find_ball(  # 调用 find_ball 方法，传入 Ball 对象和 flags 参数
            Ball(plot.x, plot.y, plot.offline, GameObjType.BALL), flags
        )

        self.interpret_results(plot, flags)  # 调用 interpret_results 方法，传入 plot 和 flags 参数

    def plot_ball(self, ball: Ball, stroke_distance: float, degrees_off: float) -> Plot:  # 定义 plot_ball 方法，接受 ball、stroke_distance 和 degrees_off 参数，返回 Plot 对象
        cup_vector = Point(0, -1)  # 创建 Point 对象 cup_vector，坐标为 (0, -1)
        rad_from_cup = math.atan2(ball.Y, ball.X) - math.atan2(  # 计算从杯子到球的弧度
            cup_vector.y, cup_vector.x
        )
        rad_from_ball = rad_from_cup - math.pi  # 计算从球到杯子的弧度

        hypotenuse = stroke_distance  # 设置变量 hypotenuse 为 stroke_distance
        adjacent = math.cos(rad_from_ball + to_radians(degrees_off)) * hypotenuse  # 计算邻边长度
        opposite = math.sqrt(math.pow(hypotenuse, 2) - math.pow(adjacent, 2))  # 计算对边长度

        new_pos: Point  # 声明变量 new_pos 为 Point 类型
        # 如果从球的位置出发，经过一定角度后的位置超过180度，则计算新位置
        if to_degrees_360(rad_from_ball + to_radians(degrees_off)) > 180:
            new_pos = Point(int(ball.X - opposite), int(ball.Y - adjacent))
        else:
            new_pos = Point(int(ball.X + opposite), int(ball.Y - adjacent))

        # 返回新位置和经过的距离
        return Plot(new_pos.x, new_pos.y, int(opposite))

    # 解释结果的方法，接受一个Plot对象和一个整数标志作为参数，不返回任何结果
    def interpret_results(self, plot: Plot, flags: int) -> None:
        # 计算球到洞的距离
        cup_distance: int = int(
            get_distance(
                Point(plot.x, plot.y),
                Point(self.hole_geometry.cup.X, self.hole_geometry.cup.Y),
            )
        )
        # 计算球到目标位置的距离
        travel_distance: int = int(
            get_distance(Point(plot.x, plot.y), Point(self.ball.X, self.ball.Y))
        )

        # 打印空行
        print(" ")
        # 检查是否击中了一杆进洞
        if (flags & ace) == ace:
            # 如果是一杆进洞，打印消息并记录成绩
            print("Hole in One! You aced it.")
            self.score_card_record_stroke(Ball(0, 0, 0, GameObjType.BALL))
            self.report_current_score()
            return

        # 检查球是否掉进了树丛中
        if (flags & in_trees) == in_trees:
            # 如果球掉进了树丛，打印消息并记录成绩，然后重新开球
            print("Your ball is lost in the trees. Take a penalty stroke.")
            self.score_card_record_stroke(self.ball)
            self.tee_up()
            return

        # 检查球是否掉进了水中
        if (flags & in_water) == in_water:
            # 如果球掉进了水中，根据50%的几率打印不同的消息，并记录成绩，然后重新开球
            if odds(50):
                msg = "Your ball has gone to a watery grave."
            else:
                msg = "Your ball is lost in the water."
            print(msg + " Take a penalty stroke.")
            self.score_card_record_stroke(self.ball)
            self.tee_up()
            return  # 返回空值，结束函数执行

        if (flags & out_of_bounds) == out_of_bounds:  # 检查是否出界标志位被设置
            print("Out of bounds. Take a penalty stroke.")  # 打印出界信息
            self.score_card_record_stroke(self.ball)  # 记录罚杆
            self.tee_up()  # 准备下一杆
            return  # 结束函数执行

        if (flags & dub) == dub:  # 检查是否挖坑标志位被设置
            print("You dubbed it.")  # 打印挖坑信息
            self.score_card_record_stroke(self.ball)  # 记录罚杆
            self.tee_up()  # 准备下一杆
            return  # 结束函数执行

        if (flags & in_cup) == in_cup:  # 检查是否进洞标志位被设置
            if odds(50):  # 以50%的概率执行
                msg = "You holed it."  # 设置信息为进洞
            else:
                msg = "It's in!"  # 设置信息为球在洞里
            print(msg)  # 打印信息
# 调用score_card_record_stroke方法，记录球的位置和类型
self.score_card_record_stroke(Ball(plot.x, plot.y, 0, GameObjType.BALL))
# 调用report_current_score方法，报告当前得分
self.report_current_score()
# 返回空值，结束函数执行
return

# 如果flags中包含slice_标志，并且不包含on_green标志
if ((flags & slice_) == slice_) and not ((flags & on_green) == on_green):
    # 如果flags中包含out_of_bounds标志
    if (flags & out_of_bounds) == out_of_bounds:
        # 设置bad为"badly"
        bad = "badly"
    else:
        # 否则设置bad为空字符串
        bad = ""
    # 打印"You sliced{bad}: {plot.offline} yards offline."
    print(f"You sliced{bad}: {plot.offline} yards offline.")

# 如果flags中包含hook标志，并且不包含on_green标志
if ((flags & hook) == hook) and not ((flags & on_green) == on_green):
    # 如果flags中包含out_of_bounds标志
    if (flags & out_of_bounds) == out_of_bounds:
        # 设置bad为"badly"
        bad = "badly"
    else:
        # 否则设置bad为空字符串
        bad = ""
    # 打印"You hooked{bad}: {plot.offline} yards offline."
    print(f"You hooked{bad}: {plot.offline} yards offline.")

# 如果self.stroke_num大于1
if self.stroke_num > 1:
    # 获取上一次击球的球对象
    prev_ball = self.score_card_get_previous_stroke()
# 计算当前球到洞的距离
d1 = get_distance(
    Point(prev_ball.X, prev_ball.Y),  # 使用前一个球的坐标创建一个点
    Point(self.hole_geometry.cup.X, self.hole_geometry.cup.Y),  # 使用洞的坐标创建一个点
)
# 获取洞到球杆的距离
d2 = cup_distance
# 如果洞到球杆的距离大于当前球到洞的距离，则打印提示信息
if d2 > d1:
    print("Too much club.")

# 如果球在粗糙地面上，则打印提示信息
if (flags & in_rough) == in_rough:
    print("You're in the rough.")

# 如果球在沙坑中，则打印提示信息
if (flags & in_sand) == in_sand:
    print("You're in a sand trap.")

# 如果球在果岭上，则根据洞到球杆的距离打印相应的提示信息
if (flags & on_green) == on_green:
    if cup_distance < 4:
        pd = str(cup_distance * 3) + " feet"  # 如果距离小于4码，则以英尺为单位
    else:
        pd = f"{cup_distance} yards"  # 如果距离大于等于4码，则以码为单位
    print(f"You're on the green. It's {pd} from the pin.")  # 打印提示信息
        # 如果球在球道上或者在草地上，打印球的飞行距离和距离洞杯的距离
        if ((flags & on_fairway) == on_fairway) or ((flags & in_rough) == in_rough):
            print(
                f"Shot went {travel_distance} yards. "
                f"It's {cup_distance} yards from the cup."
            )

        # 记录击球的一杆，并更新球的位置
        self.score_card_record_stroke(Ball(plot.x, plot.y, 0, GameObjType.BALL))

        # 将球的位置更新为当前位置
        self.ball = Ball(plot.x, plot.y, 0, GameObjType.BALL)

        # 准备下一杆击球
        self.tee_up()

    # 报告当前的比分
    def report_current_score(self) -> None:
        # 获取当前洞的标准杆数
        par = CourseInfo[self.hole_num].par
        # 如果当前洞的击球次数等于标准杆数加一，打印“柏基”，即比标准杆多一杆
        if len(self.score_card[self.hole_num]) == par + 1:
            print("A bogey. One above par.")
        # 如果当前洞的击球次数等于标准杆数，打印“标准杆”，即和标准杆一样
        if len(self.score_card[self.hole_num]) == par:
            print("Par. Nice.")
        # 如果当前洞的击球次数等于标准杆数减一，打印“柏基”，即比标准杆少一杆
        if len(self.score_card[self.hole_num]) == (par - 1):
        # 如果当前球洞的成绩等于标准杆减一，打印"A birdie! One below par."
        if len(self.score_card[self.hole_num]) == (par - 1):
            print("A birdie! One below par.")
        # 如果当前球洞的成绩等于标准杆减二，打印"An Eagle! Two below par."
        if len(self.score_card[self.hole_num]) == (par - 2):
            print("An Eagle! Two below par.")
        # 如果当前球洞的成绩等于标准杆减三，打印"Double Eagle! Unbelievable."
        if len(self.score_card[self.hole_num]) == (par - 3):
            print("Double Eagle! Unbelievable.")

        # 初始化总标准杆为0
        total_par: int = 0
        # 遍历每个球洞，累加标准杆
        for i in range(1, self.hole_num + 1):
            total_par += CourseInfo[i].par

        # 打印总标准杆和总成绩
        print(" ")
        print("-----------------------------------------------------")
        # 根据球洞数量选择"hole"或"holes"
        if self.hole_num > 1:
            hole_str = "holes"
        else:
            hole_str = "hole"
        # 打印总标准杆和总成绩
        print(
            f" Total par for {self.hole_num} {hole_str} is: {total_par}. "
            f"Your total is: {self.score_card_get_total()}."
        )
        print("-----------------------------------------------------")  # 打印分隔线
        print(" ")  # 打印空行

        if self.hole_num == 18:  # 如果当前球洞号为18
            self.game_over()  # 调用游戏结束函数
        else:  # 否则
            time.sleep(2)  # 等待2秒
            self.new_hole()  # 进入下一个球洞

    def find_ball(self, ball: Ball, flags: int) -> int:  # 定义一个函数，用于找到球的位置
        if self.is_on_fairway(ball) and not self.is_on_green(ball):  # 如果球在球道上且不在果岭上
            flags |= on_fairway  # 将标志位设置为在球道上
        if self.is_on_green(ball):  # 如果球在果岭上
            flags |= on_green  # 将标志位设置为在果岭上
        if self.is_in_rough(ball):  # 如果球在粗糙区
            flags |= in_rough  # 将标志位设置为在粗糙区
        if self.is_out_of_bounds(ball):  # 如果球出界
            flags |= out_of_bounds  # 将标志位设置为出界
        if self.is_in_hazard(ball, GameObjType.WATER):  # 如果球在危险区域（水）
            flags |= in_water  # 将标志位设置为在水中
        if self.is_in_hazard(ball, GameObjType.TREES):  # 检查球是否在树林中
            flags |= in_trees  # 如果在树林中，将标志位in_trees添加到flags中
        if self.is_in_hazard(ball, GameObjType.SAND):  # 检查球是否在沙坑中
            flags |= in_sand  # 如果在沙坑中，将标志位in_sand添加到flags中

        if ball.Y < 0:  # 检查球的Y坐标是否小于0
            flags |= passed_cup  # 如果小于0，将标志位passed_cup添加到flags中

        # 计算球到洞的距离，如果小于2，表示球在洞内
        d = get_distance(
            Point(ball.X, ball.Y),
            Point(self.hole_geometry.cup.X, self.hole_geometry.cup.Y),
        )
        if d < 2:  # 如果距离小于2
            flags |= in_cup  # 将标志位in_cup添加到flags中

        return flags  # 返回标志位

    def is_on_fairway(self, ball: Ball) -> bool:  # 检查球是否在球道上
        return is_in_rectangle(ball, self.hole_geometry.fairway)  # 返回球是否在球道矩形范围内的布尔值
    def is_on_green(self, ball: Ball) -> bool:
        # 计算球到洞的距离
        d = get_distance(
            Point(ball.X, ball.Y),
            Point(self.hole_geometry.cup.X, self.hole_geometry.cup.Y),
        )
        # 判断球是否在果岭上
        return d < self.hole_geometry.green.Radius

    def hazard_hit(self, h: Hazard, ball: Ball, hazard: GameObjType) -> bool:
        # 计算球到障碍物的距离
        d = get_distance(Point(ball.X, ball.Y), Point(h.X, h.Y))
        result = False
        # 判断球是否击中了障碍物
        if (d < h.Radius) and h.Type == hazard:
            result = True
        return result

    def is_in_hazard(self, ball: Ball, hazard: GameObjType) -> bool:
        result: bool = False
        # 遍历所有的障碍物，判断球是否在任何一个障碍物内
        for h in self.hole_geometry.hazards:
            result = result and self.hazard_hit(h, ball, hazard)
        return result
    # 检查球是否在球洞的粗糙区域内
    def is_in_rough(self, ball: Ball) -> bool:
        return is_in_rectangle(ball, self.hole_geometry.rough) and (
            not is_in_rectangle(ball, self.hole_geometry.fairway)
        )

    # 检查球是否在界外
    def is_out_of_bounds(self, ball: Ball) -> bool:
        return (not self.is_on_fairway(ball)) and (not self.is_in_rough(ball))

    # 开始新的一轮记分
    def score_card_start_new_hole(self) -> None:
        self.score_card.append([])

    # 记录击球
    def score_card_record_stroke(self, ball: Ball) -> None:
        clone = Ball(ball.X, ball.Y, 0, GameObjType.BALL)
        self.score_card[self.hole_num].append(clone)

    # 获取上一击球的信息
    def score_card_get_previous_stroke(self) -> Ball:
        return self.score_card[self.hole_num][len(self.score_card[self.hole_num]) - 1]

    # 获取总分
    def score_card_get_total(self) -> int:
        total: int = 0  # 初始化一个整型变量total，用于存储总分数
        for h in self.score_card:  # 遍历self.score_card列表中的元素
            total += len(h)  # 将每个元素的长度加到total上
        return total  # 返回总分数

    def ask(
        self, question: str, min_: int, max_: int, callback: Callable[[int], Any]
    ) -> None:
        # input from console is always an integer passed to a callback
        # or "quit" to end game
        print(question)  # 打印问题
        i = input().strip().lower()  # 从控制台获取输入并去除首尾空格，转换为小写
        if i == "quit":  # 如果输入为"quit"
            self.quit_game()  # 调用quit_game方法结束游戏
            return  # 返回
        if i == "bag":  # 如果输入为"bag"
            self.review_bag()  # 调用review_bag方法

        try:  # 尝试执行以下代码
            n = int(i)  # 将输入转换为整数
            success = True  # 初始化一个布尔变量 success 为 True
        except Exception:  # 捕获任何异常
            success = False  # 如果出现异常，将 success 设置为 False
            n = 0  # 将 n 设置为 0

        if success:  # 如果成功执行
            if n >= min_ and n <= max_:  # 如果 n 在 min_ 和 max_ 之间
                callback(n)  # 调用回调函数并传入 n
            else:  # 如果 n 不在 min_ 和 max_ 之间
                self.ask(question, min_, max_, callback)  # 调用 ask 方法并传入参数
        else:  # 如果执行不成功
            self.ask(question, min_, max_, callback)  # 调用 ask 方法并传入参数

    def review_bag(self) -> None:  # 定义一个 review_bag 方法，返回类型为 None
        print(" ")  # 打印空行
        print("  #     Club      Average Yardage")  # 打印表头
        print("-----------------------------------")  # 打印分隔线
        print("  1    Driver           250")  # 打印第一行数据
        print("  2    3 Wood           225")  # 打印第二行数据
        print("  3    5 Wood           200")  # 打印第三行数据
        print("  4    Hybrid           190")  # 打印出4号球杆的信息
        print("  5    4 Iron           170")  # 打印出5号球杆的信息
        print("  6    7 Iron           150")  # 打印出6号球杆的信息
        print("  7    9 Iron           125")  # 打印出7号球杆的信息
        print("  8    Pitching wedge   110")  # 打印出8号球杆的信息
        print("  9    Sand wedge        75")   # 打印出9号球杆的信息
        print(" 10    Putter            10")   # 打印出10号球杆的信息
        print(" ")  # 打印空行

    def quit_game(self) -> None:
        print("\nLooks like rain. Goodbye!\n")  # 打印出天气预报和道别信息
        return  # 返回空值

    def game_over(self) -> None:
        net = self.score_card_get_total() - self.handicap  # 计算净杆数
        print("Good game!")  # 打印出比赛结束的信息
        print(f"Your net score is: {net}")  # 打印出净杆数
        print("Let's visit the pro shop...")  # 打印出去专业店的信息
        print(" ")  # 打印空行
        return  # 返回空值
# 如果当前脚本被直接执行而不是被导入，则执行 Golf() 函数
# 这通常用于测试脚本的功能是否正常，或者作为脚本的入口点
# 如果脚本被导入到其他脚本中，则不会执行 Golf() 函数
```