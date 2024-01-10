# `basic-computer-games\84_Super_Star_Trek\python\superstartrek.py`

```
"""
****        **** STAR TREK ****        ****
**** SIMULATION OF A MISSION OF THE STARSHIP ENTERPRISE,
**** AS SEEN ON THE STAR TREK TV SHOW.
**** ORIGINAL PROGRAM BY MIKE MAYFIELD, MODIFIED VERSION
**** PUBLISHED IN DEC'S "101 BASIC GAMES", BY DAVE AHL.
**** MODIFICATIONS TO THE LATTER (PLUS DEBUGGING) BY BOB
**** LEEDOM - APRIL & DECEMBER 1974,
**** WITH A LITTLE HELP FROM HIS FRIENDS . . .

  Output is identical to BASIC version except for a few
  fixes (as noted, search `bug`) and minor cleanup.
"""

# 导入所需的模块
import random
import sys
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Callable, Dict, Final, List, Optional, Tuple


# 定义函数，获取用户输入的浮点数
def get_user_float(prompt: str) -> float:
    """Get input from user and return it."""
    while True:
        answer = input(prompt)
        try:
            answer_float = float(answer)
            return answer_float
        except ValueError:
            pass


# 定义枚举类型，表示不同的实体
class Entity(Enum):
    klingon = "+K+"
    ship = "<*>"
    empty = "***"
    starbase = ">!<"
    star = " * "
    void = "   "


# 定义数据类，表示二维坐标点
@dataclass
class Point:
    x: int
    y: int

    def __str__(self) -> str:
        return f"{self.x + 1} , {self.y + 1}"


# 定义数据类，表示星际象限中的位置
@dataclass
class Position:
    """
    Every quadrant has 8 sectors

    Hence the position could also be represented as:
    x = quadrant.x * 8 + sector.x
    y = quadrant.y * 8 + sector.y
    """

    quadrant: Point
    sector: Point


# 定义数据类，表示星际象限的数据
@dataclass
class QuadrantData:
    klingons: int
    bases: int
    stars: int

    def num(self) -> int:
        return 100 * self.klingons + 10 * self.bases + self.stars


# 定义数据类，表示克林贡战舰
@dataclass
class KlingonShip:
    sector: Point
    shield: float


# 定义飞船类，表示星际飞船
class Ship:
    energy_capacity: int = 3000
    torpedo_capacity: int = 10
    # 初始化方法，设置飞船的初始位置、能量、设备、损坏状态、护盾、鱼雷数量和停靠状态
    def __init__(self) -> None:
        # 设置飞船的初始位置为两个随机生成的点
        self.position = Position(Point(fnr(), fnr()), Point(fnr(), fnr()))
        # 设置飞船的初始能量为飞船的能量容量
        self.energy: int = Ship.energy_capacity
        # 设置飞船的设备列表
        self.devices: Tuple[str, ...] = (
            "WARP ENGINES",
            "SHORT RANGE SENSORS",
            "LONG RANGE SENSORS",
            "PHASER CONTROL",
            "PHOTON TUBES",
            "DAMAGE CONTROL",
            "SHIELD CONTROL",
            "LIBRARY-COMPUTER",
        )
        # 初始化飞船设备的损坏状态列表，长度为设备数量，初始值为0
        self.damage_stats: List[float] = [0] * len(self.devices)
        # 初始化飞船的护盾值为0
        self.shields = 0
        # 设置飞船的初始鱼雷数量为飞船的鱼雷容量
        self.torpedoes = Ship.torpedo_capacity
        # 初始化飞船的停靠状态为False
        self.docked: bool = False  # true when docked at starbase
    
    # 重新填充飞船的能量和鱼雷数量
    def refill(self) -> None:
        # 将飞船的能量重新设置为飞船的能量容量
        self.energy = Ship.energy_capacity
        # 将飞船的鱼雷数量重新设置为飞船的鱼雷容量
        self.torpedoes = Ship.torpedo_capacity
    
    # 调整飞船的能量和护盾值，用于导航时扣除能量
    def maneuver_energy(self, n: int) -> None:
        """Deduct the energy for navigation from energy/shields."""
        # 从飞船的能量中扣除导航所需的能量和额外的10能量
        self.energy -= n + 10
    
        # 如果飞船的能量小于等于0
        if self.energy <= 0:
            # 打印信息提示护盾控制器为导航提供能量
            print("SHIELD CONTROL SUPPLIES ENERGY TO COMPLETE THE MANEUVER.")
            # 将护盾值增加已扣除的能量
            self.shields += self.energy
            # 将飞船的能量设置为0
            self.energy = 0
            # 将护盾值设置为0或者保持不变
            self.shields = max(0, self.shields)
    def shield_control(self) -> None:
        """Raise or lower the shields."""
        # 如果护盾控制系统已经损坏，则打印信息并返回
        if self.damage_stats[6] < 0:
            print("SHIELD CONTROL INOPERABLE")
            return

        # 无限循环，直到输入有效能量值
        while True:
            energy_to_shield = input(
                f"ENERGY AVAILABLE = {self.energy + self.shields} NUMBER OF UNITS TO SHIELDS? "
            )
            if len(energy_to_shield) > 0:
                x = int(energy_to_shield)
                break

        # 如果输入值小于0或等于当前护盾能量值，则打印信息并返回
        if x < 0 or self.shields == x:
            print("<SHIELDS UNCHANGED>")
            return

        # 如果输入值大于可用能量值，则打印信息并返回
        if x > self.energy + self.shields:
            print(
                "SHIELD CONTROL REPORTS  'THIS IS NOT THE FEDERATION "
                "TREASURY.'\n"
                "<SHIELDS UNCHANGED>"
            )
            return

        # 更新能量值和护盾能量值，并打印信息
        self.energy += self.shields - x
        self.shields = x
        print("DEFLECTOR CONTROL ROOM REPORT:")
        print(f"  'SHIELDS NOW AT {self.shields} UNITS PER YOUR COMMAND.'")
class Quadrant:
    def __init__(
        self,
        point: Point,  # position of the quadrant
        population: QuadrantData,
        ship_position: Position,
    ) -> None:
        """Populate quadrant map"""
        # 确保象限点的坐标在0到7之间
        assert 0 <= point.x <= 7 and 0 <= point.y <= 7
        # 根据象限点的坐标生成象限名称
        self.name = Quadrant.quadrant_name(point.x, point.y, False)

        # 设置象限内的克林贡人口、基地和星球数量
        self.nb_klingons = population.klingons
        self.nb_bases = population.bases
        self.nb_stars = population.stars

        # 在基地修复时额外的延迟
        self.delay_in_repairs_at_base: float = 0.5 * random.random()

        # 当前象限内的克林贡舰船
        self.klingon_ships: List[KlingonShip] = []

        # 初始化空的象限地图，保存每个位置上的实体
        self.data = [[Entity.void for _ in range(8)] for _ in range(8)]

        # 填充象限地图
        self.populate_quadrant(ship_position)

    @classmethod
    def quadrant_name(cls, row: int, col: int, region_only: bool = False) -> str:
        """Return quadrant name visible on scans, etc."""
        # 定义两个区域的名称和修饰词
        region1 = [
            "ANTARES",
            "RIGEL",
            "PROCYON",
            "VEGA",
            "CANOPUS",
            "ALTAIR",
            "SAGITTARIUS",
            "POLLUX",
        ]
        region2 = [
            "SIRIUS",
            "DENEB",
            "CAPELLA",
            "BETELGEUSE",
            "ALDEBARAN",
            "REGULUS",
            "ARCTURUS",
            "SPICA",
        ]
        modifier = ["I", "II", "III", "IV"]

        # 根据行和列的值确定象限所在的区域
        quadrant = region1[row] if col < 4 else region2[row]

        # 如果不仅返回区域名称，则添加修饰词
        if not region_only:
            quadrant += " " + modifier[col % 4]

        return quadrant

    def set_value(self, x: float, y: float, entity: Entity) -> None:
        # 设置指定位置上的实体
        self.data[round(x)][round(y)] = entity

    def get_value(self, x: float, y: float) -> Entity:
        # 获取指定位置上的实体
        return self.data[round(x)][round(y)]
    # 寻找当前象限中的空位置
    def find_empty_place(self) -> Tuple[int, int]:
        """Find an empty location in the current quadrant."""
        while True:
            # 生成随机的行和列索引
            row, col = fnr(), fnr()
            # 如果该位置为空，则返回该位置的行和列索引
            if self.get_value(row, col) == Entity.void:
                return row, col

    # 填充象限
    def populate_quadrant(self, ship_position: Position) -> None:
        # 在飞船位置设置飞船实体
        self.set_value(ship_position.sector.x, ship_position.sector.y, Entity.ship)
        # 为每个克林贡飞船生成随机位置，并设置克林贡飞船实体
        for _ in range(self.nb_klingons):
            x, y = self.find_empty_place()
            self.set_value(x, y, Entity.klingon)
            # 将克林贡飞船的位置和护盾强度添加到克林贡飞船列表中
            self.klingon_ships.append(
                KlingonShip(
                    Point(x, y), klingon_shield_strength * (0.5 + random.random())
                )
            )
        # 如果存在星球基地，则在当前象限中设置星球基地实体
        if self.nb_bases > 0:
            # 在当前象限中寻找空位置来设置星球基地
            starbase_x, starbase_y = self.find_empty_place()
            self.starbase = Point(starbase_x, starbase_y)
            self.set_value(starbase_x, starbase_y, Entity.starbase)
        # 为每颗星星生成随机位置，并设置星星实体
        for _ in range(self.nb_stars):
            x, y = self.find_empty_place()
            self.set_value(x, y, Entity.star)

    # 返回象限的字符串表示
    def __str__(self) -> str:
        quadrant_string = ""
        # 遍历象限中的每一行
        for row in self.data:
            # 遍历每一行中的实体
            for entity in row:
                # 将实体的值添加到象限字符串中
                quadrant_string += entity.value
        # 返回象限字符串
        return quadrant_string
# 定义一个名为 World 的类
class World:
    # 初始化方法，设置初始值
    def __init__(
        self,
        total_klingons: int = 0,  # 游戏开始时的克林贡人数
        bases_in_galaxy: int = 0,
    # 初始化游戏环境
    def __init__(self, total_klingons: int, bases_in_galaxy: int) -> None:
        # 创建飞船对象
        self.ship = Ship()
        # 初始化星际日期
        self.initial_stardate = 100 * random.randint(20, 39)
        self.stardate: float = self.initial_stardate
        # 随机生成任务持续时间
        self.mission_duration = random.randint(25, 34)

        # 敌人
        self.remaining_klingons = total_klingons

        # 玩家星舰基地
        self.bases_in_galaxy = bases_in_galaxy

        # 初始化星系地图
        self.galaxy_map: List[List[QuadrantData]] = [
            [QuadrantData(0, 0, 0) for _ in range(8)] for _ in range(8)
        ]
        # 初始化已探索的星系地图
        self.charted_galaxy_map: List[List[QuadrantData]] = [
            [QuadrantData(0, 0, 0) for _ in range(8)] for _ in range(8)
        ]

        # 初始化星系地图内容
        for x in range(8):
            for y in range(8):
                r1 = random.random()

                if r1 > 0.98:
                    quadrant_klingons = 3
                elif r1 > 0.95:
                    quadrant_klingons = 2
                elif r1 > 0.80:
                    quadrant_klingons = 1
                else:
                    quadrant_klingons = 0
                self.remaining_klingons += quadrant_klingons

                quadrant_bases = 0
                if random.random() > 0.96:
                    quadrant_bases = 1
                    self.bases_in_galaxy += 1
                self.galaxy_map[x][y] = QuadrantData(
                    quadrant_klingons, quadrant_bases, 1 + fnr()
                )

        # 如果剩余克林贡人数大于任务持续时间，则更新任务持续时间
        if self.remaining_klingons > self.mission_duration:
            self.mission_duration = self.remaining_klingons + 1

        # 如果星系中没有基地，则在飞船所在的象限添加一个基地
        if self.bases_in_galaxy == 0:  # original has buggy extra code here
            self.bases_in_galaxy = 1
            self.galaxy_map[self.ship.position.quadrant.x][
                self.ship.position.quadrant.y
            ].bases += 1

        # 初始化当前象限
        curr = self.ship.position.quadrant
        self.quadrant = Quadrant(
            self.ship.position.quadrant,
            self.galaxy_map[curr.x][curr.y],
            self.ship.position,
        )
    # 返回任务剩余时间，即初始星日期加上任务持续时间减去当前星日期
    def remaining_time(self) -> float:
        return self.initial_stardate + self.mission_duration - self.stardate
    
    # 判断任务是否已经结束，即剩余时间是否小于0
    def has_mission_ended(self) -> bool:
        return self.remaining_time() < 0
class Game:
    """Handle user actions"""

    def __init__(self) -> None:
        # 初始化 restart 属性为 False
        self.restart = False
        # 创建 World 对象并赋值给 self.world 属性
        self.world = World()

    def startup(self) -> None:
        """Initialize the game variables and map, and print startup messages."""
        # 打印游戏启动画面
        print(
            "\n\n\n\n\n\n\n\n\n\n\n"
            "                                    ,------*------,\n"
            "                    ,-------------   '---  ------'\n"
            "                     '-------- --'      / /\n"
            "                         ,---' '-------/ /--,\n"
            "                          '----------------'\n\n"
            "                    THE USS ENTERPRISE --- NCC-1701\n"
            "\n\n\n\n"
        )
        # 获取 self.world 属性的值
        world = self.world
        # 打印游戏初始信息
        print(
            "YOUR ORDERS ARE AS FOLLOWS:\n"
            f"     DESTROY THE {world.remaining_klingons} KLINGON WARSHIPS WHICH HAVE INVADED\n"
            "   THE GALAXY BEFORE THEY CAN ATTACK FEDERATION HEADQUARTERS\n"
            f"   ON STARDATE {world.initial_stardate+world.mission_duration}. "
            f" THIS GIVES YOU {world.mission_duration} DAYS. THERE "
            f"{'IS' if world.bases_in_galaxy == 1 else 'ARE'}\n"
            f"   {world.bases_in_galaxy} "
            f"STARBASE{'' if world.bases_in_galaxy == 1 else 'S'} IN THE GALAXY FOR "
            "RESUPPLYING YOUR SHIP.\n"
        )
    # 定义一个新的象限方法，没有返回值
    def new_quadrant(self) -> None:
        """Enter a new quadrant: populate map and print a short range scan."""
        # 获取世界和飞船对象
        world = self.world
        ship = world.ship
        # 获取飞船所在象限
        q = ship.position.quadrant

        # 创建一个新的象限对象，包括象限坐标、星系地图信息和飞船位置
        world.quadrant = Quadrant(
            q,
            world.galaxy_map[q.x][q.y],
            ship.position,
        )

        # 将星系地图中该象限标记为已探索
        world.charted_galaxy_map[q.x][q.y] = world.galaxy_map[q.x][q.y]

        # 如果当前星时等于初始星时，打印初始位置信息
        if world.stardate == world.initial_stardate:
            print("\nYOUR MISSION BEGINS WITH YOUR STARSHIP LOCATED")
            print(f"IN THE GALACTIC QUADRANT, '{world.quadrant.name}'.\n")
        else:
            # 否则打印进入象限信息
            print(f"\nNOW ENTERING {world.quadrant.name} QUADRANT . . .\n")

        # 如果象限内有克林贡战舰，打印战斗状态，并检查护盾状态
        if world.quadrant.nb_klingons != 0:
            print("COMBAT AREA      CONDITION RED")
            if ship.shields <= 200:
                print("   SHIELDS DANGEROUSLY LOW")
        # 调用短程扫描方法
        self.short_range_scan()

    # 定义一个方法，返回企业号和第i艘克林贡战舰之间的距离
    def fnd(self, i: int) -> float:
        """Find distance between Enterprise and i'th Klingon warship."""
        # 获取飞船和第i艘克林贡战舰的位置
        ship = self.world.ship.position.sector
        klingons = self.world.quadrant.klingon_ships[i].sector
        # 计算距离并返回
        return sqrt((klingons.x - ship.x) ** 2 + (klingons.y - ship.y) ** 2)
    def klingons_fire(self) -> None:
        """处理附近克林贡人对企业的攻击。"""
        # 获取世界中的飞船对象
        ship = self.world.ship

        # 如果当前象限内没有克林贡人，则返回
        if self.world.quadrant.nb_klingons <= 0:
            return
        # 如果飞船停靠，则打印信息并返回
        if ship.docked:
            print("STARBASE SHIELDS PROTECT THE ENTERPRISE")
            return

        # 遍历当前象限内的克林贡飞船
        for i, klingon_ship in enumerate(self.world.quadrant.klingon_ships):
            # 如果克林贡飞船的护盾值小于等于0，则继续下一次循环
            if klingon_ship.shield <= 0:
                continue

            # 计算克林贡飞船对企业的伤害值
            h = int((klingon_ship.shield / self.fnd(i)) * (random.random() + 2))
            # 减少企业的护盾值
            ship.shields -= h
            # 减少克林贡飞船的护盾值
            klingon_ship.shield /= random.random() + 3
            # 打印企业受到的伤害信息
            print(f" {h} UNIT HIT ON ENTERPRISE FROM SECTOR {klingon_ship.sector} ")
            # 如果企业的护盾值小于等于0，则游戏结束，企业被摧毁
            if ship.shields <= 0:
                self.end_game(won=False, quit=False, enterprise_killed=True)
                return
            # 打印企业护盾值减少后的信息
            print(f"      <SHIELDS DOWN TO {ship.shields} UNITS>")
            # 如果伤害值大于等于20且随机数小于0.60且伤害值与企业护盾值的比值大于0.02，则执行以下代码
            if h >= 20 and random.random() < 0.60 and h / ship.shields > 0.02:
                # 随机选择一个设备
                device = fnr()
                # 减少企业受损设备的伤害值
                ship.damage_stats[device] -= h / ship.shields + 0.5 * random.random()
                # 打印受损设备的信息
                print(
                    f"DAMAGE CONTROL REPORTS  '{ship.devices[device]} DAMAGED BY THE HIT'"
                )

    def long_range_scan(self) -> None:
        """打印远程扫描结果。"""
        # 如果企业的长程传感器受损，则打印信息并返回
        if self.world.ship.damage_stats[2] < 0:
            print("LONG RANGE SENSORS ARE INOPERABLE")
            return

        # 打印当前象限的远程扫描结果
        print(f"LONG RANGE SCAN FOR QUADRANT {self.world.ship.position.quadrant}")
        print_scan_results(
            self.world.ship.position.quadrant,
            self.world.galaxy_map,
            self.world.charted_galaxy_map,
        )
    def damage_control(self) -> None:
        """Print a damage control report."""
        # 获取世界中的飞船对象
        ship = self.world.ship

        # 如果飞船的第五项损坏统计小于0，则打印报告不可用
        if ship.damage_stats[5] < 0:
            print("DAMAGE CONTROL REPORT NOT AVAILABLE")
        else:
            # 打印设备的状态修复情况
            print("\nDEVICE             STATE OF REPAIR")
            for r1 in range(8):
                print(
                    f"{ship.devices[r1].ljust(26, ' ')}{int(ship.damage_stats[r1] * 100) * 0.01:g}"
                )
            print()

        # 如果飞船没有停靠，则返回
        if not ship.docked:
            return

        # 计算损坏设备的总和
        damage_sum = sum(0.1 for i in range(8) if ship.damage_stats[i] < 0)
        # 如果总和为0，则返回
        if damage_sum == 0:
            return

        # 将损坏总和加上基地修复的延迟时间
        damage_sum += self.world.quadrant.delay_in_repairs_at_base
        # 如果损坏总和大于等于1，则将其设为0.9
        if damage_sum >= 1:
            damage_sum = 0.9
        # 打印修复飞船所需的估计时间
        print("\nTECHNICIANS STANDING BY TO EFFECT REPAIRS TO YOUR SHIP;")
        print(
            f"ESTIMATED TIME TO REPAIR: {round(0.01 * int(100 * damage_sum), 2)} STARDATES"
        )
        # 如果用户不授权修复命令，则返回
        if input("WILL YOU AUTHORIZE THE REPAIR ORDER (Y/N)? ").upper().strip() != "Y":
            return

        # 对于每个损坏的设备，将其修复状态设为0
        for i in range(8):
            if ship.damage_stats[i] < 0:
                ship.damage_stats[i] = 0
        # 更新星期时间
        self.world.stardate += damage_sum + 0.1

    def end_game(
        self, won: bool = False, quit: bool = True, enterprise_killed: bool = False
    ) -> None:
        """Handle end-of-game situations."""
        # 处理游戏结束的情况
        if won:
            # 如果游戏胜利
            print("CONGRATULATIONS, CAPTAIN! THE LAST KLINGON BATTLE CRUISER")
            print("MENACING THE FEDERATION HAS BEEN DESTROYED.\n")
            print(
                f"YOUR EFFICIENCY RATING IS {round(1000 * (self.world.remaining_klingons / (self.world.stardate - self.world.initial_stardate))**2, 4)}\n\n"
            )
        else:
            # 如果游戏失败
            if not quit:
                # 如果没有退出游戏
                if enterprise_killed:
                    print(
                        "\nTHE ENTERPRISE HAS BEEN DESTROYED. THE FEDERATION "
                        "WILL BE CONQUERED."
                    )
                print(f"IT IS STARDATE {round(self.world.stardate, 1)}")

            print(
                f"THERE WERE {self.world.remaining_klingons} KLINGON BATTLE CRUISERS LEFT AT"
            )
            print("THE END OF YOUR MISSION.\n\n")

            if self.world.bases_in_galaxy == 0:
                # 如果银河系中没有基地
                sys.exit()

        print("THE FEDERATION IS IN NEED OF A NEW STARSHIP COMMANDER")
        print("FOR A SIMILAR MISSION -- IF THERE IS A VOLUNTEER,")
        if input("LET HIM STEP FORWARD AND ENTER 'AYE'? ").upper().strip() != "AYE":
            # 如果没有人愿意接受新任务
            sys.exit()
        self.restart = True
# 克林贡护盾强度设定为200
klingon_shield_strength: Final = 200
# 8个扇区等于1个象限
dirs: Final = [  # (down-up, left,right)
    [0, 1],  # 1: 向右移动 (同 #9)
    [-1, 1],  # 2: 向右上移动
    [-1, 0],  # 3: 向上移动 (x 坐标减小；北方)
    [-1, -1],  # 4: 向左上移动 (西北方)
    [0, -1],  # 5: 向左移动 (西方)
    [1, -1],  # 6: 向左下移动 (西南方)
    [1, 0],  # 7: 向下移动 (x 坐标增大；南方)
    [1, 1],  # 8: 向右下移动
    [0, 1],  # 9: 向右移动 (东方)
]  # 基本方向向量


def fnr() -> int:
    """生成一个从0到7的随机整数（包括0和7）。"""
    return random.randint(0, 7)


def print_scan_results(
    quadrant: Point,
    galaxy_map: List[List[QuadrantData]],
    charted_galaxy_map: List[List[QuadrantData]],
) -> None:
    sep = "-------------------"
    print(sep)
    for x in (quadrant.x - 1, quadrant.x, quadrant.x + 1):
        n: List[Optional[int]] = [None, None, None]

        # 揭示当前地图的部分
        for y in (quadrant.y - 1, quadrant.y, quadrant.y + 1):
            if 0 <= x <= 7 and 0 <= y <= 7:
                n[y - quadrant.y + 1] = galaxy_map[x][y].num()
                charted_galaxy_map[x][y] = galaxy_map[x][y]

        line = ": "
        for line_col in n:
            if line_col is None:
                line += "*** : "
            else:
                line += str(line_col + 1000).rjust(4, " ")[-3:] + " : "
        print(line)
        print(sep)


def print_direction(source: Point, to: Point) -> None:
    """打印网格中两个位置之间的方向和距离。"""
    delta1 = -(to.x - source.x)  # 翻转，使得正数表示向上移动（方向 = 3）
    delta2 = to.y - source.y

    if delta2 > 0:
        if delta1 < 0:
            base = 7
        else:
            base = 1
            delta1, delta2 = delta2, delta1
    else:
        if delta1 > 0:
            base = 3
        else:
            base = 5
            delta1, delta2 = delta2, delta1

    delta1, delta2 = abs(delta1), abs(delta2)
    # 如果 delta1 或 delta2 大于 0，则执行以下代码；原始代码中存在 bug，没有检查是否会除以 0
    if delta1 > 0 or delta2 > 0:  
        # 如果 delta1 大于等于 delta2，则计算并打印 DIRECTION
        if delta1 >= delta2:
            print(f"DIRECTION = {round(base + delta2 / delta1, 6)}")
        # 否则计算并打印 DIRECTION
        else:
            print(f"DIRECTION = {round(base + 2 - delta1 / delta2, 6)}")

    # 计算并打印 DISTANCE
    print(f"DISTANCE = {round(sqrt(delta1 ** 2 + delta2 ** 2), 6)}")
# 定义主函数，不返回任何结果
def main() -> None:
    # 创建游戏对象
    game = Game()
    # 获取游戏世界对象
    world = game.world
    # 获取世界中的飞船对象
    ship = world.ship

    # 创建命令字典，将命令映射到游戏对象的方法
    f: Dict[str, Callable[[], None]] = {
        "NAV": game.navigation,
        "SRS": game.short_range_scan,
        "LRS": game.long_range_scan,
        "PHA": game.phaser_control,
        "TOR": game.photon_torpedoes,
        "SHE": ship.shield_control,
        "DAM": game.damage_control,
        "COM": game.computer,
        "XXX": game.end_game,
    }

    # 游戏循环
    while True:
        # 游戏初始化
        game.startup()
        # 创建新的象限
        game.new_quadrant()
        restart = False

        # 内部循环，直到游戏重新开始
        while not restart:
            # 检查飞船能量和护盾是否低于阈值，如果是则输出错误信息
            if ship.shields + ship.energy <= 10 or (
                ship.energy <= 10 and ship.damage_stats[6] != 0
            ):
                print(
                    "\n** FATAL ERROR **   YOU'VE JUST STRANDED YOUR SHIP "
                    "IN SPACE.\nYOU HAVE INSUFFICIENT MANEUVERING ENERGY, "
                    "AND SHIELD CONTROL\nIS PRESENTLY INCAPABLE OF CROSS-"
                    "CIRCUITING TO ENGINE ROOM!!"
                )

            # 获取用户输入的命令
            command = input("COMMAND? ").upper().strip()

            # 如果命令在命令字典中，则执行对应的游戏方法
            if command in f:
                f[command]()
            else:
                # 如果命令不在命令字典中，则输出错误信息
                print(
                    "ENTER ONE OF THE FOLLOWING:\n"
                    "  NAV  (TO SET COURSE)\n"
                    "  SRS  (FOR SHORT RANGE SENSOR SCAN)\n"
                    "  LRS  (FOR LONG RANGE SENSOR SCAN)\n"
                    "  PHA  (TO FIRE PHASERS)\n"
                    "  TOR  (TO FIRE PHOTON TORPEDOES)\n"
                    "  SHE  (TO RAISE OR LOWER SHIELDS)\n"
                    "  DAM  (FOR DAMAGE CONTROL REPORTS)\n"
                    "  COM  (TO CALL ON LIBRARY-COMPUTER)\n"
                    "  XXX  (TO RESIGN YOUR COMMAND)\n"
                )


if __name__ == "__main__":
    main()
```