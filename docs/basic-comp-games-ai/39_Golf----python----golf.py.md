# `basic-computer-games\39_Golf\python\golf.py`

```
# 定义了一个 ASCII 艺术风格的高尔夫球场图案
'''
Despite being a text based game, the code uses simple geometry to simulate a course.
# 尽管是基于文本的游戏，但代码使用简单的几何图形来模拟高尔夫球场。
Fairways are 40 yard wide rectangles, surrounded by 5 yards of rough around the perimeter.
# 球道是宽40码的矩形，周围有5码的粗糙区域。
The green is a circle of 10 yards radius around the cup.
# 球洞周围是半径为10码的圆形绿地。
The cup is always at point (0,0).
# 球洞始终在点(0,0)处。

Using basic trigonometry we can plot the ball's location using the distance of the stroke and
and the angle of deviation (hook/slice).
# 使用基本的三角学，我们可以根据击球的距离和偏离角度（勾钩/切球）来绘制球的位置。

The stroke distances are based on real world averages of different club types.
# 击球距离基于不同球杆类型的真实世界平均值。
Lots of randomization, "business rules", and luck influence the game play.
# 大量的随机化、“业务规则”和运气影响了游戏的玩法。
Probabilities are commented in the code.
# 概率在代码中有注释。

note: 'courseInfo', 'clubs', & 'scoreCard' arrays each include an empty object so indexing
can begin at 1. Like all good programmers we count from zero, but in this context,
it's more natural when hole number one is at index one
# 注意：'courseInfo'、'clubs'和'scoreCard'数组每个都包含一个空对象，以便从1开始索引。像所有优秀的程序员一样，我们从零开始计数，但在这种情况下，当第一个洞的编号为1时，从1开始更自然。

    |-----------------------------|
    |            rough            |
    |   ----------------------    |
    |   |                     |   |
    | r |        =  =         | r |
    | o |     =        =      | o |
    | u |    =    .     =     | u |
    | g |    =   green  =     | g |
    | h |     =        =      | h |
    |   |        =  =         |   |
    |   |                     |   |
    |   |                     |   |
    |   |      Fairway        |   |
    |   |                     |   |
    |   |               ------    |
    |   |            --        -- |
    |   |           --  hazard  --|
    |   |            --        -- |
    |   |               ------    |
    |   |                     |   |
    |   |                     |   |
    |   |                     |   |   out
    |   |                     |   |   of
    |   |                     |   |   bounds
    |   |                     |   |
    |   |                     |   |
    |            tee              |
# 典型的果岭尺寸：20-30码
# 典型的高尔夫球场球道宽度为35到45码
# 我们的球道延伸5码超出果岭
# 我们的草地是球道周围5码的边界

# 根据球的位置、击球距离和偏离线度（hook或slice）计算球的新位置

# 对于右撇子：
# 切：正度数=球向右
# 钩：负度数=球向左

# 杯子始终在点：0,0。
# 我们使用 atan2 来计算杯子和球之间的角度。
# 将杯子的向量设置为0，-1在360度圆上等同于：
# 0度=12点钟；90度=3点钟；180度=6点钟；270=9点钟
# 杯子和球之间的反向角度是 PI 的差异（使用弧度）。

# 给定角度和击球距离（斜边），我们使用余弦来计算三角形的对边和邻边，这就是球的新位置。

# 杯子
# |
# |
# | 对边
# |-----* 新位置
# |    /
# |   /
# 邻边 |  /
# | /  斜边
# |/
# 出发点

# <- 钩    切 ->

# 鉴于需要描述特定击球/球位置的大量组合，
# 我们使用“位掩码”技术来描述击球结果。
# 使用位掩码，多个标志（位）被组合成一个可以通过应用掩码来测试的单个二进制数。
# 掩码是另一个二进制数，它隔离了您感兴趣的特定位。
# 然后，您可以应用您的语言的位运算符来测试或设置标志。

# 游戏设计：Jason Bonthron，2021
# www.bonthron.com
# 献给我父亲，狂热的高尔夫球手 Raymond Bonthron

# 灵感来自于1978年的“高尔夫”游戏，出自“Basic Computer Games”作者 Steve North
# 他修改了一位未知作者的现有高尔夫游戏

# 2022年由 Martin Thoma 移植到 Python
# 导入必要的模块
from functools import partial
from typing import Any, Callable, List, NamedTuple, Tuple

# 清空控制台的函数
def clear_console() -> None:
    print("\033[H\033[J", end="")

# 定义一个包含 x 和 y 坐标的点的类
class Point(NamedTuple):
    x: int
    y: int

# 定义游戏对象的类型枚举
class GameObjType(enum.Enum):
    BALL = enum.auto()
    CUP = enum.auto()
    GREEN = enum.auto()
    FAIRWAY = enum.auto()
    ROUGH = enum.auto()
    TREES = enum.auto()
    WATER = enum.auto()
    SAND = enum.auto()

# 定义一个圆形游戏对象的类
class CircleGameObj(NamedTuple):
    # 中心点
    X: int
    Y: int
    Radius: int
    Type: GameObjType

# 定义一个矩形游戏对象的类
class RectGameObj(NamedTuple):
    # 左上角坐标
    X: int
    Y: int
    Width: int
    Length: int
    Type: GameObjType

# 别名定义
Ball = CircleGameObj
Hazard = CircleGameObj

# 定义一个包含球洞信息的类
class HoleInfo(NamedTuple):
    hole: int
    yards: int
    par: int
    hazards: List[Hazard]
    description: str

# 定义一个包含球洞几何信息的类
class HoleGeometry(NamedTuple):
    cup: CircleGameObj
    green: CircleGameObj
    fairway: RectGameObj
    rough: RectGameObj
    hazards: List[Hazard]

# 定义一个包含绘图信息的类
@dataclass
class Plot:
    x: int
    y: int
    offline: int

# 计算两点之间的距离
def get_distance(pt1: Point, pt2: Point) -> float:
    """distance between 2 points"""
    return math.sqrt(math.pow((pt2.x - pt1.x), 2) + math.pow((pt2.y - pt1.y), 2))

# 判断一个圆形游戏对象是否在矩形游戏对象内部
def is_in_rectangle(pt: CircleGameObj, rect: RectGameObj) -> bool:
    # 只有当完全在内部时返回 True
    return (
        (pt.X > rect.X)
        and (pt.X < rect.X + rect.Width)
        and (pt.Y > rect.Y)
        and (pt.Y < rect.Y + rect.Length)
    )

# 角度转弧度
def to_radians(angle: float) -> float:
    return angle * (math.pi / 180.0)

# 弧度转360度角度
def to_degrees_360(angle: float) -> float:
    """radians to 360 degrees"""
    deg = angle * (180.0 / math.pi)
    if deg < 0.0:
        deg += 360.0
    return deg

# 计算一个整数小于等于给定参数的概率
def odds(x: int) -> bool:
    # 1-100 之间的整数是否小于等于给定参数
    return random.randint(1, 101) <= x

# 球场信息
CourseInfo = [
    HoleInfo(0, 0, 0, [], ""),  # 包含一个空白项，使得索引 1 == 球洞 1
    # -------------------------------------------------------- front 9
    # 创建 HoleInfo 对象，表示第1个洞的信息
    HoleInfo(
        1,  # 洞号
        361,  # 码数
        4,  # 标准杆数
        [  # 障碍物列表
            Hazard(20, 100, 10, GameObjType.TREES),  # 树木障碍物
            Hazard(-20, 80, 10, GameObjType.TREES),  # 树木障碍物
            Hazard(-20, 100, 10, GameObjType.TREES),  # 树木障碍物
        ],
        "There are a couple of trees on the left and right.",  # 描述信息
    ),
    # 创建 HoleInfo 对象，表示第2个洞的信息
    HoleInfo(
        2,
        389,
        4,
        [Hazard(0, 160, 20, GameObjType.WATER)],  # 水障碍物
        "There is a large water hazard across the fairway about 150 yards.",  # 描述信息
    ),
    # 创建 HoleInfo 对象，表示第3个洞的信息
    HoleInfo(
        3,
        206,
        3,
        [  # 障碍物列表
            Hazard(20, 20, 5, GameObjType.WATER),  # 水障碍物
            Hazard(-20, 160, 10, GameObjType.WATER),  # 水障碍物
            Hazard(10, 12, 5, GameObjType.SAND),  # 沙坑障碍物
        ],
        "There is some sand and water near the green.",  # 描述信息
    ),
    # 创建 HoleInfo 对象，表示第4个洞的信息
    HoleInfo(
        4,
        500,
        5,
        [Hazard(-14, 12, 12, GameObjType.SAND)],  # 沙坑障碍物
        "There's a bunker to the left of the green.",  # 描述信息
    ),
    # 创建 HoleInfo 对象，表示第5个洞的信息
    HoleInfo(
        5,
        408,
        4,
        [  # 障碍物列表
            Hazard(20, 120, 20, GameObjType.TREES),  # 树木障碍物
            Hazard(20, 160, 20, GameObjType.TREES),  # 树木障碍物
            Hazard(10, 20, 5, GameObjType.SAND),  # 沙坑障碍物
        ],
        "There are some trees to your right.",  # 描述信息
    ),
    # 创建 HoleInfo 对象，表示第6个洞的信息
    HoleInfo(
        6,
        359,
        4,
        [Hazard(14, 0, 4, GameObjType.SAND), Hazard(-14, 0, 4, GameObjType.SAND)],  # 沙坑障碍物
        "",  # 描述信息为空
    ),
    # 创建 HoleInfo 对象，表示第7个洞的信息
    HoleInfo(
        7,
        424,
        5,
        [  # 障碍物列表
            Hazard(20, 200, 10, GameObjType.SAND),  # 沙坑障碍物
            Hazard(10, 180, 10, GameObjType.SAND),  # 沙坑障碍物
            Hazard(20, 160, 10, GameObjType.SAND),  # 沙坑障碍物
        ],
        "There are several sand traps along your right.",  # 描述信息
    ),
    # 创建 HoleInfo 对象，表示第8个洞的信息
    HoleInfo(8, 388, 4, [Hazard(-20, 340, 10, GameObjType.TREES)], ""),  # 描述信息为空
    # 创建 HoleInfo 对象，表示第9个洞的信息
    HoleInfo(
        9,
        196,
        3,
        [  # 障碍物列表
            Hazard(-30, 180, 20, GameObjType.TREES),  # 树木障碍物
            Hazard(14, -8, 5, GameObjType.SAND),  # 沙坑障碍物
        ],
        "",  # 描述信息为空
    ),
    # -------------------------------------------------------- back 9
    # 第10洞的信息
    HoleInfo(
        hole=10,  # 洞号
        yards=400,  # 码数
        par=4,  # 标准杆数
        hazards=[  # 障碍物列表
            Hazard(-14, -8, 5, GameObjType.SAND),  # 沙坑位置和大小
            Hazard(14, -8, 5, GameObjType.SAND),  # 沙坑位置和大小
        ],
        description="",  # 描述
    ),
    # 第11洞的信息
    HoleInfo(
        11,  # 洞号
        560,  # 码数
        5,  # 标准杆数
        [  # 障碍物列表
            Hazard(-20, 400, 10, GameObjType.TREES),  # 树木位置和大小
            Hazard(-10, 380, 10, GameObjType.TREES),  # 树木位置和大小
            Hazard(-20, 260, 10, GameObjType.TREES),  # 树木位置和大小
            Hazard(-20, 200, 10, GameObjType.TREES),  # 树木位置和大小
            Hazard(-10, 180, 10, GameObjType.TREES),  # 树木位置和大小
            Hazard(-20, 160, 10, GameObjType.TREES),  # 树木位置和大小
        ],
        "Lots of trees along the left of the fairway.",  # 描述
    ),
    # 第12洞的信息
    HoleInfo(
        12,  # 洞号
        132,  # 码数
        3,  # 标准杆数
        [  # 障碍物列表
            Hazard(-10, 120, 10, GameObjType.WATER),  # 水域位置和大小
            Hazard(-5, 100, 10, GameObjType.SAND),  # 沙坑位置和大小
        ],
        "There is water and sand directly in front of you. A good drive should clear both.",  # 描述
    ),
    # 第13洞的信息
    HoleInfo(
        13,  # 洞号
        357,  # 码数
        4,  # 标准杆数
        [  # 障碍物列表
            Hazard(-20, 200, 10, GameObjType.TREES),  # 树木位置和大小
            Hazard(-10, 180, 10, GameObjType.TREES),  # 树木位置和大小
            Hazard(-20, 160, 10, GameObjType.TREES),  # 树木位置和大小
            Hazard(14, 12, 8, GameObjType.SAND),  # 沙坑位置和大小
        ],
        "",  # 描述
    ),
    # 第14洞的信息
    HoleInfo(14, 294, 4, [Hazard(0, 20, 10, GameObjType.SAND)], ""),  # 洞号、码数、标准杆数、障碍物列表、描述
    # 第15洞的信息
    HoleInfo(
        15,  # 洞号
        475,  # 码数
        5,  # 标准杆数
        [  # 障碍物列表
            Hazard(-20, 20, 10, GameObjType.WATER),  # 水域位置和大小
            Hazard(10, 20, 10, GameObjType.SAND),  # 沙坑位置和大小
        ],
        "Some sand and water near the green.",  # 描述
    ),
    # 第16洞的信息
    HoleInfo(16, 375, 4, [Hazard(-14, -8, 5, GameObjType.SAND)], ""),  # 洞号、码数、标准杆数、障碍物列表、描述
    # 第17洞的信息
    HoleInfo(
        17,  # 洞号
        180,  # 码数
        3,  # 标准杆数
        [  # 障碍物列表
            Hazard(20, 100, 10, GameObjType.TREES),  # 树木位置和大小
            Hazard(-20, 80, 10, GameObjType.TREES),  # 树木位置和大小
        ],
        "",  # 描述
    ),
    # 第18洞的信息
    HoleInfo(
        18,  # 洞号
        550,  # 码数
        5,  # 标准杆数
        [Hazard(20, 30, 15, GameObjType.WATER)],  # 水域位置和大小
        "There is a water hazard near the green.",  # 描述
    ), 
# -------------------------------------------------------- bitwise Flags
# 定义各种标志位，用于表示不同的状态
dub = 0b00000000000001
hook = 0b00000000000010
slice_ = 0b00000000000100
passed_cup = 0b00000000001000
in_cup = 0b00000000010000
on_fairway = 0b00000000100000
on_green = 0b00000001000000
in_rough = 0b00000010000000
in_sand = 0b00000100000000
in_trees = 0b00001000000000
in_water = 0b00010000000000
out_of_bounds = 0b00100000000000
luck = 0b01000000000000
ace = 0b10000000000000

# 定义 Golf 类
class Golf:
    # 定义 Golf 类的属性
    ball: Ball
    hole_num: int = 0
    stroke_num: int = 0
    handicap: int = 0
    player_difficulty: int = 0
    hole_geometry: HoleGeometry

    # 定义球道的宽度、延伸长度和粗糙区域的大小
    fairway_width: int = 40
    fairway_extension: int = 5
    rough_amt: int = 5

    # 记录每一杆击球后球的位置
    score_card: List[List[Ball]] = [[]]

    # 球杆清单，包括名称和平均击球距离
    clubs: List[Tuple[str, int]] = [
        ("", 0),
        ("Driver", 250),
        ("3 Wood", 225),
        ("5 Wood", 200),
        ("Hybrid", 190),
        ("4 Iron", 170),
        ("7 Iron", 150),
        ("9 Iron", 125),
        ("Pitching wedge", 110),
        ("Sand wedge", 75),
        ("Putter", 10),
    ]
    # 初始化方法，打印欢迎信息和游戏规则，然后等待用户按下任意键继续
    def __init__(self) -> None:
        print(" ")
        print('          8""""8 8"""88 8     8"""" ')
        print('          8    " 8    8 8     8     ')
        print("          8e     8    8 8e    8eeee ")
        print("          88  ee 8    8 88    88    ")
        print("          88   8 8    8 88    88    ")
        print("          88eee8 8eeee8 88eee 88    ")
        print(" ")
        print("Welcome to the Creative Computing Country Club,")
        print("an eighteen hole championship layout located a short")
        print("distance from scenic downtown Lambertville, New Jersey.")
        print("The game will be explained as you play.")
        print("Enjoy your game! See you at the 19th hole...")
        print(" ")
        print("Type QUIT at any time to leave the game.")
        print("Type BAG at any time to review the clubs in your bag.")
        print(" ")

        # 等待用户按下任意键继续，然后清空控制台并开始游戏
        input("Press any key to continue.")
        clear_console()
        self.start_game()

    # 开始游戏方法，打印用户的球杆信息，并等待用户按下任意键继续
    def start_game(self) -> None:
        print(" ")
        print("              YOUR BAG")
        self.review_bag()
        print("Type BAG at any time to review the clubs in your bag.")
        print(" ")

        # 等待用户按下任意键继续，然后清空控制台并询问用户的球技等级
        input("Press any key to continue.")
        clear_console()
        self.ask_handicap()

    # 询问用户的球技等级方法，等待用户输入并将输入限制在0到30之间
    def ask_handicap(self) -> None:
        print(" ")

        self.ask(
            "PGA handicaps range from 0 to 30.\nWhat is your handicap?",
            0,
            30,
            self.set_handicap_ask_difficulty,
        )

    # 设置用户的球技等级并询问用户最差的球技问题
    def set_handicap_ask_difficulty(self, i: int) -> None:
        self.handicap = i
        print(" ")

        self.ask(
            (
                "Common difficulties at golf include:\n"
                "1=Hook, 2=Slice, 3=Poor Distance, 4=Trap Shots, 5=Putting\n"
                "Which one is your worst?"
            ),
            1,
            5,
            self.set_difficulty_and_hole,
        )
    # 设置球员的难度和洞号
    def set_difficulty_and_hole(self, j: int) -> None:
        # 设置球员的难度
        self.player_difficulty = j
        # 清空控制台
        clear_console()
        # 创建新的洞
        self.new_hole()

    # 创建新的洞
    def new_hole(self) -> None:
        # 洞号加一
        self.hole_num += 1
        # 击球数归零
        self.stroke_num = 0

        # 获取当前洞的信息
        info: HoleInfo = CourseInfo[self.hole_num]

        # 获取当前洞的码数
        yards: int = info.yards
        # 从发球台到洞的距离
        cup = CircleGameObj(0, 0, 0, GameObjType.CUP)
        green = CircleGameObj(0, 0, 10, GameObjType.GREEN)

        # 创建球道对象
        fairway = RectGameObj(
            0 - int(self.fairway_width / 2),
            0 - (green.Radius + self.fairway_extension),
            self.fairway_width,
            yards + (green.Radius + self.fairway_extension) + 1,
            GameObjType.FAIRWAY,
        )

        # 创建粗糙地形对象
        rough = RectGameObj(
            fairway.X - self.rough_amt,
            fairway.Y - self.rough_amt,
            fairway.Width + (2 * self.rough_amt),
            fairway.Length + (2 * self.rough_amt),
            GameObjType.ROUGH,
        )

        # 创建球对象
        self.ball = Ball(0, yards, 0, GameObjType.BALL)

        # 开始新的洞的记分卡
        self.score_card_start_new_hole()

        # 设置当前洞的几何信息
        self.hole_geometry = HoleGeometry(cup, green, fairway, rough, info.hazards)

        # 打印当前洞的信息
        print(f"                |> {self.hole_num}")
        print("                |        ")
        print("                |        ")
        print("          ^^^^^^^^^^^^^^^")

        print(
            f"Hole #{self.hole_num}. You are at the tee. Distance {info.yards} yards, par {info.par}."
        )
        print(info.description)

        # 准备发球
        self.tee_up()

    # 设置推杆和击球
    def set_putter_and_stroke(self, strength: float) -> None:
        # 获取推杆
        putter = self.clubs[self.putt]
        # 进行击球
        self.stroke((putter[1] * (strength / 10.0)), self.putt)
    # 询问用户选择球杆，并设置当前球杆
    def ask_gauge(self, c: int) -> None:
        # 设置当前球杆为用户选择的球杆
        self.club = self.clubs[c]

        # 打印当前球杆的平均击球距离
        print(" ")
        print(f"[{self.club[0].upper()}: average {self.club[1]} yards]")

        # 创建一个部分函数，用于设置当前击球距离
        foo = partial(self.make_stroke, c=c)

        # 询问用户选择击球距离的百分比，并调用部分函数
        self.ask(
            "Now gauge your distance by a percentage of a full swing. (1-10)",
            1,
            10,
            foo,
        )

    # 进行击球动作
    def make_stroke(self, strength: float, c: int) -> None:
        # 根据用户选择的击球强度和当前球杆的平均击球距离进行击球
        self.stroke((self.club[1] * (strength / 10.0)), c)

    # 准备开球
    def tee_up(self) -> None:
        # 如果在果岭上，自动选择推杆
        # 否则询问用户选择球杆和击球强度
        if self.is_on_green(self.ball) and not self.is_in_hazard(
            self.ball, GameObjType.SAND
        ):
            # 设置当前球杆为推杆
            self.putt = 10
            print("[PUTTER: average 10 yards]")
            # 根据概率生成提示信息
            if odds(20):
                msg = "Keep your head down.\n"
            else:
                msg = ""

            # 询问用户选择推杆击球强度
            self.ask(
                msg + "Choose your putt potency. (1-10)",
                1,
                10,
                self.set_putter_and_stroke,
            )
        else:
            # 询问用户选择球杆
            self.ask("What club do you choose? (1-10)", 1, 10, self.ask_gauge)
    # 根据球的位置、击球距离和偏离角度绘制球的轨迹
    def plot_ball(self, ball: Ball, stroke_distance: float, degrees_off: float) -> Plot:
        # 创建一个指向杯子的向量
        cup_vector = Point(0, -1)
        # 计算球相对于杯子的角度
        rad_from_cup = math.atan2(ball.Y, ball.X) - math.atan2(
            cup_vector.y, cup_vector.x
        )
        # 计算相对于球的角度
        rad_from_ball = rad_from_cup - math.pi

        # 计算斜边、邻边和对边
        hypotenuse = stroke_distance
        adjacent = math.cos(rad_from_ball + to_radians(degrees_off)) * hypotenuse
        opposite = math.sqrt(math.pow(hypotenuse, 2) - math.pow(adjacent, 2))

        new_pos: Point
        # 根据角度判断新位置的坐标
        if to_degrees_360(rad_from_ball + to_radians(degrees_off)) > 180:
            new_pos = Point(int(ball.X - opposite), int(ball.Y - adjacent))
        else:
            new_pos = Point(int(ball.X + opposite), int(ball.Y - adjacent))

        # 返回新位置和对边的长度
        return Plot(new_pos.x, new_pos.y, int(opposite))
    # 报告当前比分情况的方法，不返回任何内容
    def report_current_score(self) -> None:
        # 获取当前球洞的标准杆数
        par = CourseInfo[self.hole_num].par
        # 如果当前球洞的成绩长度等于标准杆数加一，打印“柏忌。比标准杆多一杆。”
        if len(self.score_card[self.hole_num]) == par + 1:
            print("A bogey. One above par.")
        # 如果当前球洞的成绩长度等于标准杆数，打印“标准杆。不错。”
        if len(self.score_card[self.hole_num]) == par:
            print("Par. Nice.")
        # 如果当前球洞的成绩长度等于标准杆数减一，打印“小鸟！比标准杆少一杆。”
        if len(self.score_card[self.hole_num]) == (par - 1):
            print("A birdie! One below par.")
        # 如果当前球洞的成绩长度等于标准杆数减二，打印“老鹰！比标准杆少两杆。”
        if len(self.score_card[self.hole_num]) == (par - 2):
            print("An Eagle! Two below par.")
        # 如果当前球洞的成绩长度等于标准杆数减三，打印“双老鹰！难以置信。”
        if len(self.score_card[self.hole_num]) == (par - 3):
            print("Double Eagle! Unbelievable.")

        # 初始化总标准杆数为0
        total_par: int = 0
        # 遍历所有球洞，累加标准杆数
        for i in range(1, self.hole_num + 1):
            total_par += CourseInfo[i].par

        # 打印空行和分隔线
        print(" ")
        print("-----------------------------------------------------")
        # 根据球洞数量选择“hole”或“holes”
        if self.hole_num > 1:
            hole_str = "holes"
        else:
            hole_str = "hole"
        # 打印总标准杆数和玩家总成绩
        print(
            f" Total par for {self.hole_num} {hole_str} is: {total_par}. "
            f"Your total is: {self.score_card_get_total()}."
        )
        # 打印分隔线和空行
        print("-----------------------------------------------------")
        print(" ")

        # 如果当前球洞为18，游戏结束，否则等待2秒后进入下一球洞
        if self.hole_num == 18:
            self.game_over()
        else:
            time.sleep(2)
            self.new_hole()
    # 查找球的位置并更新标志位
    def find_ball(self, ball: Ball, flags: int) -> int:
        # 如果球在球道上且不在果岭上，将标志位设置为在球道上
        if self.is_on_fairway(ball) and not self.is_on_green(ball):
            flags |= on_fairway
        # 如果球在果岭上，将标志位设置为在果岭上
        if self.is_on_green(ball):
            flags |= on_green
        # 如果球在粗糙区域，将标志位设置为在粗糙区域
        if self.is_in_rough(ball):
            flags |= in_rough
        # 如果球出界，将标志位设置为出界
        if self.is_out_of_bounds(ball):
            flags |= out_of_bounds
        # 如果球在水障碍中，将标志位设置为在水中
        if self.is_in_hazard(ball, GameObjType.WATER):
            flags |= in_water
        # 如果球在树障碍中，将标志位设置为在树林中
        if self.is_in_hazard(ball, GameObjType.TREES):
            flags |= in_trees
        # 如果球在沙坑中，将标志位设置为在沙坑中
        if self.is_in_hazard(ball, GameObjType.SAND):
            flags |= in_sand

        # 如果球的 Y 坐标小于 0，将标志位设置为通过洞口
        if ball.Y < 0:
            flags |= passed_cup

        # 计算球到洞口的距离，如果小于 2，将标志位设置为在洞内
        d = get_distance(
            Point(ball.X, ball.Y),
            Point(self.hole_geometry.cup.X, self.hole_geometry.cup.Y),
        )
        if d < 2:
            flags |= in_cup

        # 返回更新后的标志位
        return flags

    # 判断球是否在球道上
    def is_on_fairway(self, ball: Ball) -> bool:
        return is_in_rectangle(ball, self.hole_geometry.fairway)

    # 判断球是否在果岭上
    def is_on_green(self, ball: Ball) -> bool:
        # 计算球到洞口的距离，判断是否小于果岭半径
        d = get_distance(
            Point(ball.X, ball.Y),
            Point(self.hole_geometry.cup.X, self.hole_geometry.cup.Y),
        )
        return d < self.hole_geometry.green.Radius

    # 判断球是否击中障碍物
    def hazard_hit(self, h: Hazard, ball: Ball, hazard: GameObjType) -> bool:
        # 计算球到障碍物的距离
        d = get_distance(Point(ball.X, ball.Y), Point(h.X, h.Y))
        result = False
        # 如果距离小于障碍物半径且障碍物类型与指定类型相同，返回 True
        if (d < h.Radius) and h.Type == hazard:
            result = True
        return result

    # 判断球是否在指定类型的障碍物中
    def is_in_hazard(self, ball: Ball, hazard: GameObjType) -> bool:
        result: bool = False
        # 遍历所有障碍物，判断球是否在指定类型的障碍物中
        for h in self.hole_geometry.hazards:
            result = result and self.hazard_hit(h, ball, hazard)
        return result

    # 判断球是否在粗糙区域中
    def is_in_rough(self, ball: Ball) -> bool:
        # 判断球是否在粗糙区域中且不在球道上
        return is_in_rectangle(ball, self.hole_geometry.rough) and (
            not is_in_rectangle(ball, self.hole_geometry.fairway)
        )
    # 检查球是否超出界限，如果不在球道上且不在草地上，则返回 True
    def is_out_of_bounds(self, ball: Ball) -> bool:
        return (not self.is_on_fairway(ball)) and (not self.is_in_rough(ball))

    # 开始新的一轮记分
    def score_card_start_new_hole(self) -> None:
        self.score_card.append([])

    # 记录击球次数
    def score_card_record_stroke(self, ball: Ball) -> None:
        clone = Ball(ball.X, ball.Y, 0, GameObjType.BALL)
        self.score_card[self.hole_num].append(clone)

    # 获取上一次击球的球
    def score_card_get_previous_stroke(self) -> Ball:
        return self.score_card[self.hole_num][len(self.score_card[self.hole_num]) - 1]

    # 获取总击球次数
    def score_card_get_total(self) -> int:
        total: int = 0
        for h in self.score_card:
            total += len(h)
        return total

    # 提问函数，根据输入的问题、最小值、最大值和回调函数进行交互
    def ask(
        self, question: str, min_: int, max_: int, callback: Callable[[int], Any]
    ) -> None:
        # 从控制台获取输入，输入为整数则传递给回调函数，输入为"quit"则结束游戏
        print(question)
        i = input().strip().lower()
        if i == "quit":
            self.quit_game()
            return
        if i == "bag":
            self.review_bag()

        try:
            n = int(i)
            success = True
        except Exception:
            success = False
            n = 0

        if success:
            # 如果输入的数字在最小值和最大值之间，则传递给回调函数，否则重新提问
            if n >= min_ and n <= max_:
                callback(n)
            else:
                self.ask(question, min_, max_, callback)
        else:
            # 如果输入不是整数，则重新提问
            self.ask(question, min_, max_, callback)
    # 打印球杆清单，包括编号、名称和平均码数
    def review_bag(self) -> None:
        # 打印表头
        print(" ")
        print("  #     Club      Average Yardage")
        print("-----------------------------------")
        # 打印每个球杆的编号、名称和平均码数
        print("  1    Driver           250")
        print("  2    3 Wood           225")
        print("  3    5 Wood           200")
        print("  4    Hybrid           190")
        print("  5    4 Iron           170")
        print("  6    7 Iron           150")
        print("  7    9 Iron           125")
        print("  8    Pitching wedge   110")
        print("  9    Sand wedge        75")
        print(" 10    Putter            10")
        print(" ")
    
    # 打印天气不好时的退出信息
    def quit_game(self) -> None:
        print("\nLooks like rain. Goodbye!\n")
        return
    
    # 打印比赛结束时的总结信息
    def game_over(self) -> None:
        # 计算净得分
        net = self.score_card_get_total() - self.handicap
        # 打印比赛结束信息和净得分
        print("Good game!")
        print(f"Your net score is: {net}")
        print("Let's visit the pro shop...")
        print(" ")
        return
# 如果当前模块被直接执行，则调用 Golf 函数
if __name__ == "__main__":
    Golf()
```