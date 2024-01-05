# `84_Super_Star_Trek\python\superstartrek.py`

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
import random  # 导入随机数模块
import sys  # 导入系统模块
from dataclasses import dataclass  # 导入数据类模块
from enum import Enum  # 导入枚举模块
from math import sqrt  # 导入数学模块中的平方根函数
from typing import Callable, Dict, Final, List, Optional, Tuple
# 导入需要的类型提示模块

def get_user_float(prompt: str) -> float:
    """Get input from user and return it."""
    # 定义一个函数，提示用户输入并返回输入的浮点数
    while True:
        answer = input(prompt)
        try:
            answer_float = float(answer)
            return answer_float
        except ValueError:
            pass

# 定义一个枚举类型 Entity，包含不同实体的表示
class Entity(Enum):
    klingon = "+K+"
    ship = "<*>"
    empty = "***"
    starbase = ">!<"
    star = " * "
    void = "   "  # 创建一个字符串变量 void，赋值为空格字符串

@dataclass  # 使用 dataclass 装饰器来定义一个数据类
class Point:  # 定义一个名为 Point 的类
    x: int  # 类属性 x，表示整数类型
    y: int  # 类属性 y，表示整数类型

    def __str__(self) -> str:  # 定义一个返回字符串类型的方法
        return f"{self.x + 1} , {self.y + 1}"  # 返回 x 和 y 属性值加 1 的字符串表示形式

@dataclass  # 使用 dataclass 装饰器来定义一个数据类
class Position:  # 定义一个名为 Position 的类
    """
    Every quadrant has 8 sectors

    Hence the position could also be represented as:
    x = quadrant.x * 8 + sector.x
    y = quadrant.y * 8 + sector.y
    """  # 类的文档字符串，解释了位置的表示方法
    """

    # 定义一个名为 Point 的类，表示坐标点
    quadrant: Point
    sector: Point

# 使用 dataclass 装饰器定义一个名为 QuadrantData 的数据类，包含 klingons、bases、stars 三个属性
@dataclass
class QuadrantData:
    klingons: int
    bases: int
    stars: int

    # 定义一个名为 num 的方法，返回 klingons、bases、stars 三个属性的加权和
    def num(self) -> int:
        return 100 * self.klingons + 10 * self.bases + self.stars

# 使用 dataclass 装饰器定义一个名为 KlingonShip 的数据类，包含 sector、shield 两个属性
@dataclass
class KlingonShip:
    sector: Point
    shield: float
class Ship:
    # 定义船只的能量容量
    energy_capacity: int = 3000
    # 定义船只的鱼雷容量
    torpedo_capacity: int = 10

    def __init__(self) -> None:
        # 初始化船只的位置
        self.position = Position(Point(fnr(), fnr()), Point(fnr(), fnr()))
        # 初始化船只的能量为最大能量容量
        self.energy: int = Ship.energy_capacity
        # 初始化船只的设备列表
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
        # 初始化船只的设备损坏统计列表
        self.damage_stats: List[float] = [0] * len(self.devices)
        self.shields = 0  # 初始化护盾能量为0
        self.torpedoes = Ship.torpedo_capacity  # 初始化鱼雷数量为船只的鱼雷容量
        self.docked: bool = False  # true when docked at starbase  # 当停靠在星际基地时，docked为True

    def refill(self) -> None:
        self.energy = Ship.energy_capacity  # 将能量充满到船只的能量容量
        self.torpedoes = Ship.torpedo_capacity  # 将鱼雷数量充满到船只的鱼雷容量

    def maneuver_energy(self, n: int) -> None:
        """Deduct the energy for navigation from energy/shields."""
        self.energy -= n + 10  # 从能量中扣除用于导航的能量

        if self.energy <= 0:  # 如果能量小于等于0
            print("SHIELD CONTROL SUPPLIES ENERGY TO COMPLETE THE MANEUVER.")  # 打印信息
            self.shields += self.energy  # 将能量转移到护盾
            self.energy = 0  # 能量清零
            self.shields = max(0, self.shields)  # 确保护盾能量不为负数

    def shield_control(self) -> None:
        """Raise or lower the shields."""  # 提高或降低护盾
        # 如果第6个元素的值小于0，则打印“SHIELD CONTROL INOPERABLE”并返回
        if self.damage_stats[6] < 0:
            print("SHIELD CONTROL INOPERABLE")
            return

        # 无限循环，直到用户输入能量转移到护盾的数量
        while True:
            energy_to_shield = input(
                f"ENERGY AVAILABLE = {self.energy + self.shields} NUMBER OF UNITS TO SHIELDS? "
            )
            # 如果用户输入的能量数量大于0，则将其转换为整数并跳出循环
            if len(energy_to_shield) > 0:
                x = int(energy_to_shield)
                break

        # 如果用户输入的能量数量小于0或等于当前护盾能量数量，则打印"<SHIELDS UNCHANGED>"并返回
        if x < 0 or self.shields == x:
            print("<SHIELDS UNCHANGED>")
            return

        # 如果用户输入的能量数量大于当前能量加上护盾能量的总和，则打印相应信息
        if x > self.energy + self.shields:
            print(
                "SHIELD CONTROL REPORTS  'THIS IS NOT THE FEDERATION "
                "TREASURY.'\n"
# 定义一个类 Quadrant，表示星舰游戏中的一个象限
class Quadrant:
    # 初始化方法，接受象限的位置、人口数据和飞船位置作为参数
    def __init__(
        self,
        point: Point,  # 象限的位置
        population: QuadrantData,  # 人口数据
        ship_position: Position,  # 飞船位置
    ) -> None:
        """Populate quadrant map"""
        # 使用断言确保象限的位置在合法范围内
        assert 0 <= point.x <= 7 and 0 <= point.y <= 7
        # 设置象限的名称
        self.name = Quadrant.quadrant_name(point.x, point.y, False)  # 象限名称的计算方法
        self.nb_klingons = population.klingons  # 从population对象中获取克林贡人口数量
        self.nb_bases = population.bases  # 从population对象中获取基地数量
        self.nb_stars = population.stars  # 从population对象中获取星球数量

        # 在基地修复时额外延迟
        self.delay_in_repairs_at_base: float = 0.5 * random.random()  # 设置基地修复时的额外延迟时间

        # 当前象限内的克林贡飞船
        self.klingon_ships: List[KlingonShip] = []  # 初始化一个空的克林贡飞船列表

        # 初始化空的数据列表，保存每个位置上的实体
        self.data = [[Entity.void for _ in range(8)] for _ in range(8)]  # 创建一个8x8的空数据列表

        self.populate_quadrant(ship_position)  # 调用populate_quadrant方法，填充象限

    @classmethod
    def quadrant_name(cls, row: int, col: int, region_only: bool = False) -> str:
        """Return quadrant name visible on scans, etc."""
        region1 = [  # 定义region1列表
# 创建一个包含星座名称的列表
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
# 创建另一个包含星座名称的列表
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
# 创建一个包含星座修饰词的列表
modifier = ["I", "II", "III", "IV"]
        quadrant = region1[row] if col < 4 else region2[row]  # 根据列数判断所属的区块，选择对应的区块名称

        if not region_only:  # 如果不仅仅是区块名称
            quadrant += " " + modifier[col % 4]  # 在区块名称后面添加修饰词
        return quadrant  # 返回区块名称

    def set_value(self, x: float, y: float, entity: Entity) -> None:  # 设置指定位置的实体值
        self.data[round(x)][round(y)] = entity  # 将指定位置的实体值设置为给定的实体

    def get_value(self, x: float, y: float) -> Entity:  # 获取指定位置的实体值
        return self.data[round(x)][round(y)]  # 返回指定位置的实体值

    def find_empty_place(self) -> Tuple[int, int]:  # 查找当前区块中的空位置
        """Find an empty location in the current quadrant."""  # 查找当前区块中的空位置
        while True:  # 循环直到找到空位置
            row, col = fnr(), fnr()  # 随机生成行列数
            if self.get_value(row, col) == Entity.void:  # 如果指定位置的实体值为空
                return row, col  # 返回该位置的行列数
    # 在给定位置放置飞船，并在该位置设置实体值为飞船
    def populate_quadrant(self, ship_position: Position) -> None:
        self.set_value(ship_position.sector.x, ship_position.sector.y, Entity.ship)
        # 在象限内生成指定数量的克林贡飞船
        for _ in range(self.nb_klingons):
            x, y = self.find_empty_place()  # 找到空位置
            self.set_value(x, y, Entity.klingon)  # 在该位置设置实体值为克林贡飞船
            # 创建克林贡飞船对象并添加到克林贡飞船列表中
            self.klingon_ships.append(
                KlingonShip(
                    Point(x, y), klingon_shield_strength * (0.5 + random.random())
                )
            )
        # 如果星球基地数量大于0
        if self.nb_bases > 0:
            # 在当前象限内找到空位置放置星球基地
            starbase_x, starbase_y = self.find_empty_place()
            self.starbase = Point(starbase_x, starbase_y)  # 设置星球基地的位置
            self.set_value(starbase_x, starbase_y, Entity.starbase)  # 在该位置设置实体值为星球基地
        # 在象限内生成指定数量的星球
        for _ in range(self.nb_stars):
            x, y = self.find_empty_place()  # 找到空位置
            self.set_value(x, y, Entity.star)  # 在该位置设置实体值为星球
    def __str__(self) -> str:
        # 定义一个空字符串，用于存储四象限的数据
        quadrant_string = ""
        # 遍历四象限的数据
        for row in self.data:
            for entity in row:
                # 将每个实体的值添加到字符串中
                quadrant_string += entity.value
        # 返回四象限的数据字符串
        return quadrant_string


class World:
    def __init__(
        self,
        total_klingons: int = 0,  # 游戏开始时的克林贡人数
        bases_in_galaxy: int = 0,
    ) -> None:
        # 创建一个飞船对象
        self.ship = Ship()
        # 随机生成初始星际日期
        self.initial_stardate = 100 * random.randint(20, 39)
        # 将初始星际日期赋值给当前星际日期
        self.stardate: float = self.initial_stardate
        # 随机生成任务持续时间
        self.mission_duration = random.randint(25, 34)

        # 敌人
        # 剩余克林贡人数
        self.remaining_klingons = total_klingons

        # 玩家星舰基地在星系中的位置
        self.bases_in_galaxy = bases_in_galaxy

        # 创建一个8x8的星系地图，每个位置都有一个QuadrantData对象
        self.galaxy_map: List[List[QuadrantData]] = [
            [QuadrantData(0, 0, 0) for _ in range(8)] for _ in range(8)
        ]

        # 创建一个已探索的星系地图，每个位置都有一个QuadrantData对象
        self.charted_galaxy_map: List[List[QuadrantData]] = [
            [QuadrantData(0, 0, 0) for _ in range(8)] for _ in range(8)
        ]

        # 初始化星系地图的内容
        for x in range(8):
            for y in range(8):
                r1 = random.random()

                # 根据随机数r1的大小确定当前象限的克林贡人数
                if r1 > 0.98:
                    quadrant_klingons = 3
                elif r1 > 0.95:
                quadrant_klingons = 2  # 设置象限克林贡人的数量为2
                elif r1 > 0.80:  # 如果r1大于0.80
                    quadrant_klingons = 1  # 设置象限克林贡人的数量为1
                else:  # 否则
                    quadrant_klingons = 0  # 设置象限克林贡人的数量为0
                self.remaining_klingons += quadrant_klingons  # 将象限克林贡人的数量加到剩余克林贡人数量上

                quadrant_bases = 0  # 设置象限基地的数量为0
                if random.random() > 0.96:  # 如果随机数大于0.96
                    quadrant_bases = 1  # 设置象限基地的数量为1
                    self.bases_in_galaxy += 1  # 将银河系中的基地数量加1
                self.galaxy_map[x][y] = QuadrantData(  # 在银河地图的特定位置设置象限数据
                    quadrant_klingons, quadrant_bases, 1 + fnr()  # 包括象限克林贡人数量、象限基地数量和一个随机数
                )

        if self.remaining_klingons > self.mission_duration:  # 如果剩余克林贡人数量大于任务持续时间
            self.mission_duration = self.remaining_klingons + 1  # 将任务持续时间设置为剩余克林贡人数量加1

        if self.bases_in_galaxy == 0:  # 如果银河系中的基地数量为0（原始代码在这里有错误的额外代码）
            self.bases_in_galaxy = 1  # 将银河系中的基地数量设置为1
            self.galaxy_map[self.ship.position.quadrant.x][
                self.ship.position.quadrant.y
            ].bases += 1
```
将飞船所在的象限的基地数量加一。

```
        curr = self.ship.position.quadrant
        self.quadrant = Quadrant(
            self.ship.position.quadrant,
            self.galaxy_map[curr.x][curr.y],
            self.ship.position,
        )
```
将当前飞船所在的象限信息和位置信息传递给Quadrant类的实例。

```
    def remaining_time(self) -> float:
        return self.initial_stardate + self.mission_duration - self.stardate
```
计算剩余时间，返回浮点数。

```
    def has_mission_ended(self) -> bool:
        return self.remaining_time() < 0
```
判断任务是否已经结束，返回布尔值。

```
class Game:
    """Handle user actions"""
```
Game类，用于处理用户的操作。
    def __init__(self) -> None:
        # 初始化 restart 变量为 False
        self.restart = False
        # 创建 World 对象并赋值给 self.world
        self.world = World()

    def startup(self) -> None:
        """初始化游戏变量和地图，并打印启动消息。"""
        # 打印启动消息
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
        # 将 self.world 赋值给局部变量 world
        world = self.world
        # 打印指令消息
        print(
            "YOUR ORDERS ARE AS FOLLOWS:\n"
            f"     DESTROY THE {world.remaining_klingons} KLINGON WARSHIPS WHICH HAVE INVADED\n"  # 使用world.remaining_klingons变量插入字符串中，显示剩余克林贡战舰数量
            "   THE GALAXY BEFORE THEY CAN ATTACK FEDERATION HEADQUARTERS\n"  # 显示警告信息
            f"   ON STARDATE {world.initial_stardate+world.mission_duration}. "  # 使用world.initial_stardate和world.mission_duration变量插入字符串中，显示星际日期和任务持续时间
            f" THIS GIVES YOU {world.mission_duration} DAYS. THERE "  # 使用world.mission_duration变量插入字符串中，显示任务持续时间
            f"{'IS' if world.bases_in_galaxy == 1 else 'ARE'}\n"  # 根据world.bases_in_galaxy的值选择显示IS或ARE
            f"   {world.bases_in_galaxy} "  # 使用world.bases_in_galaxy变量插入字符串中，显示星际基地数量
            f"STARBASE{'' if world.bases_in_galaxy == 1 else 'S'} IN THE GALAXY FOR "  # 根据world.bases_in_galaxy的值选择显示STARBASE或STARBASES
            "RESUPPLYING YOUR SHIP.\n"  # 显示信息

    def new_quadrant(self) -> None:
        """Enter a new quadrant: populate map and print a short range scan."""  # 进入新的象限：填充地图并打印短程扫描
        world = self.world  # 将self.world赋值给world变量
        ship = world.ship  # 将world.ship赋值给ship变量
        q = ship.position.quadrant  # 获取飞船位置的象限

        world.quadrant = Quadrant(  # 创建一个新的Quadrant对象
            q,  # 传入象限参数
            world.galaxy_map[q.x][q.y],  # 传入星系地图中对应象限的数据
            ship.position,  # 传入飞船位置
        )

        world.charted_galaxy_map[q.x][q.y] = world.galaxy_map[q.x][q.y]  # 将星际图中的特定坐标的值赋给已探索的星际图中对应的坐标

        if world.stardate == world.initial_stardate:  # 如果当前星际日期等于初始星际日期
            print("\nYOUR MISSION BEGINS WITH YOUR STARSHIP LOCATED")  # 打印任务开始的提示信息
            print(f"IN THE GALACTIC QUADRANT, '{world.quadrant.name}'.\n")  # 打印星系象限的名称
        else:
            print(f"\nNOW ENTERING {world.quadrant.name} QUADRANT . . .\n")  # 打印进入星系象限的提示信息

        if world.quadrant.nb_klingons != 0:  # 如果星系象限中克林贡战舰的数量不为0
            print("COMBAT AREA      CONDITION RED")  # 打印战斗区域的警告信息
            if ship.shields <= 200:  # 如果星舰的护盾值小于等于200
                print("   SHIELDS DANGEROUSLY LOW")  # 打印护盾值过低的警告信息
        self.short_range_scan()  # 调用short_range_scan方法进行短程扫描

    def fnd(self, i: int) -> float:
        """Find distance between Enterprise and i'th Klingon warship."""
        ship = self.world.ship.position.sector  # 获取星舰的位置
        klingons = self.world.quadrant.klingon_ships[i].sector  # 获取第i艘克林贡战舰的位置
        return sqrt((klingons.x - ship.x) ** 2 + (klingons.y - ship.y) ** 2)
        # 返回两点之间的距离，使用了欧几里得距离公式

    def klingons_fire(self) -> None:
        """Process nearby Klingons firing on Enterprise."""
        # 处理附近克林贡人对企业号的开火行为

        ship = self.world.ship
        # 获取世界中的飞船对象

        if self.world.quadrant.nb_klingons <= 0:
            return
        # 如果当前象限中没有克林贡人，则返回

        if ship.docked:
            print("STARBASE SHIELDS PROTECT THE ENTERPRISE")
            return
        # 如果飞船停靠在星舰基地，则输出信息并返回

        for i, klingon_ship in enumerate(self.world.quadrant.klingon_ships):
            if klingon_ship.shield <= 0:
                continue
            # 如果克林贡人的护盾已经耗尽，则跳过当前克林贡人

            h = int((klingon_ship.shield / self.fnd(i)) * (random.random() + 2))
            # 计算克林贡人对企业号造成的伤害
            ship.shields -= h
            # 减少企业号的护盾值
            klingon_ship.shield /= random.random() + 3
            # 减少克林贡人的护盾值
            print(f" {h} UNIT HIT ON ENTERPRISE FROM SECTOR {klingon_ship.sector} ")
            # 输出企业号受到的伤害信息
            if ship.shields <= 0:  # 如果飞船的护盾值小于等于0
                self.end_game(won=False, quit=False, enterprise_killed=True)  # 调用end_game方法，传入参数表示游戏失败，不是退出游戏，企业被摧毁
                return  # 返回
            print(f"      <SHIELDS DOWN TO {ship.shields} UNITS>")  # 打印飞船护盾值下降到多少单位
            if h >= 20 and random.random() < 0.60 and h / ship.shields > 0.02:  # 如果敌方攻击力大于等于20，并且随机数小于0.60，并且敌方攻击力除以飞船护盾值大于0.02
                device = fnr()  # 调用fnr函数，获取设备
                ship.damage_stats[device] -= h / ship.shields + 0.5 * random.random()  # 飞船受到的伤害等于敌方攻击力除以飞船护盾值加上0.5乘以一个随机数
                print(
                    f"DAMAGE CONTROL REPORTS  '{ship.devices[device]} DAMAGED BY THE HIT'"
                )  # 打印受到攻击的设备受损报告

    def phaser_control(self) -> None:  # 定义phaser_control方法，没有返回值
        """Take phaser control input and fire phasers."""  # 获取相位控制输入并发射相位炮
        world = self.world  # 获取世界对象
        klingon_ships = world.quadrant.klingon_ships  # 获取克林贡飞船列表
        ship = world.ship  # 获取飞船对象

        if ship.damage_stats[3] < 0:  # 如果飞船的第3个设备受损程度小于0
            print("PHASERS INOPERATIVE")  # 打印相位炮不可用
            return  # 返回
        # 如果当前象限内没有克林贡战舰，则打印无敌舰船的信息并返回
        if self.world.quadrant.nb_klingons <= 0:
            print("SCIENCE OFFICER SPOCK REPORTS  'SENSORS SHOW NO ENEMY SHIPS")
            print("                                IN THIS QUADRANT'")
            return

        # 如果飞船的第7项损坏统计小于0，则打印计算机故障信息
        if ship.damage_stats[7] < 0:
            print("COMPUTER FAILURE HAMPERS ACCURACY")

        # 打印激光炮锁定目标的信息以及可用能量单位
        print(f"PHASERS LOCKED ON TARGET;  ENERGY AVAILABLE = {ship.energy} UNITS")
        phaser_firepower: float = 0
        while True:
            while True:
                # 请求用户输入要开火的能量单位
                units_to_fire = input("NUMBER OF UNITS TO FIRE? ")
                if len(units_to_fire) > 0:
                    phaser_firepower = int(units_to_fire)
                    break
            # 如果输入的能量单位小于等于0，则返回
            if phaser_firepower <= 0:
                return
            # 如果飞船的能量大于等于要开火的能量单位，则继续执行后续操作
            if ship.energy >= phaser_firepower:
                break  # 结束当前循环，跳出循环体
            print(f"ENERGY AVAILABLE = {ship.energy} UNITS")  # 打印飞船剩余能量

        ship.energy -= phaser_firepower  # 减去激光炮的能量消耗
        if ship.damage_stats[7] < 0:  # 检查飞船的损坏状态，修复原始代码中的错误
            phaser_firepower *= random.random()  # 根据随机数调整激光炮火力

        phaser_per_klingon = int(phaser_firepower / self.world.quadrant.nb_klingons)  # 计算每艘克林贡飞船受到的激光炮火力
        for i, klingon_ship in enumerate(klingon_ships):  # 遍历克林贡飞船列表
            if klingon_ship.shield <= 0:  # 如果克林贡飞船护盾已经损坏
                continue  # 跳过当前循环，继续下一次循环

            h = int((phaser_per_klingon / self.fnd(i)) * (random.random() + 2))  # 计算对克林贡飞船造成的伤害
            if h <= 0.15 * klingon_ship.shield:  # 如果伤害小于克林贡飞船护盾的15%
                print(f"SENSORS SHOW NO DAMAGE TO ENEMY AT {klingon_ship.sector}")  # 打印未对克林贡飞船造成伤害的信息
            else:
                klingon_ship.shield -= h  # 减去克林贡飞船的护盾值
                print(f" {h} UNIT HIT ON KLINGON AT SECTOR {klingon_ship.sector}")  # 打印对克林贡飞船造成伤害的信息
                if klingon_ship.shield <= 0:  # 如果克林贡飞船护盾值小于等于0
                    print("*** KLINGON DESTROYED ***")  # 打印克林贡飞船被摧毁的信息
                    # 减少当前象限的克林贡数量
                    self.world.quadrant.nb_klingons -= 1
                    # 减少整个星系中的克林贡数量
                    world.remaining_klingons -= 1
                    # 在当前象限中设置克林贡飞船所在的位置为虚空
                    world.quadrant.set_value(
                        klingon_ship.sector.x, klingon_ship.sector.y, Entity.void
                    )
                    # 将克林贡飞船的护盾设为0
                    klingon_ship.shield = 0
                    # 在星系地图中减少当前飞船所在象限的克林贡数量
                    world.galaxy_map[ship.position.quadrant.x][
                        ship.position.quadrant.y
                    ].klingons -= 1
                    # 将当前飞船所在象限的星系地图标记为已探索
                    world.charted_galaxy_map[ship.position.quadrant.x][
                        ship.position.quadrant.y
                    ] = world.galaxy_map[ship.position.quadrant.x][
                        ship.position.quadrant.y
                    ]
                    # 如果剩余克林贡数量小于等于0，则结束游戏并胜利
                    if world.remaining_klingons <= 0:
                        self.end_game(won=True, quit=False)
                        return
                else:
                    # 打印传感器显示的克林贡飞船护盾剩余单位数
                    print(
                        f"   (SENSORS SHOW {round(klingon_ship.shield,6)} UNITS REMAINING)"
    def photon_torpedoes(self) -> None:
        """Take photon torpedo input and process firing of torpedoes."""
        # 获取世界对象
        world = self.world
        # 获取克林贡飞船列表
        klingon_ships = world.quadrant.klingon_ships
        # 获取飞船对象
        ship = world.ship

        # 如果鱼雷数量小于等于0，则输出提示信息并返回
        if ship.torpedoes <= 0:
            print("ALL PHOTON TORPEDOES EXPENDED")
            return
        # 如果飞船的第五项损伤状态小于0，则输出提示信息并返回
        if ship.damage_stats[4] < 0:
            print("PHOTON TUBES ARE NOT OPERATIONAL")
            return

        # 获取用户输入的鱼雷发射方向
        cd = get_user_float("PHOTON TORPEDO COURSE (1-9)? ")
        # 如果用户输入的方向为9，则将其设置为1
        if cd == 9:
            cd = 1
        if cd < 1 or cd >= 9:  # 如果cd小于1或者大于等于9
            print("ENSIGN CHEKOV REPORTS, 'INCORRECT COURSE DATA, SIR!'")  # 打印错误信息
            return  # 返回

        cdi = int(cd)  # 将cd转换为整数类型

        # Interpolate direction: 插值方向
        dx = dirs[cdi - 1][0] + (dirs[cdi][0] - dirs[cdi - 1][0]) * (cd - cdi)  # 计算x方向的插值
        dy = dirs[cdi - 1][1] + (dirs[cdi][1] - dirs[cdi - 1][1]) * (cd - cdi)  # 计算y方向的插值

        ship.energy -= 2  # 船的能量减少2
        ship.torpedoes -= 1  # 船的鱼雷数量减少1

        # Exact position
        x: float = ship.position.sector.x  # 获取船的精确x坐标
        y: float = ship.position.sector.y  # 获取船的精确y坐标

        # Rounded position (to coordinates)
        torpedo_x, torpedo_y = x, y  # 将x和y坐标赋值给鱼雷的x和y坐标
        print("TORPEDO TRACK:")  # 打印鱼雷轨迹
        while True:  # 进入无限循环，直到满足条件跳出循环
            x += dx  # 根据给定的增量更新 x 坐标
            y += dy  # 根据给定的增量更新 y 坐标
            torpedo_x, torpedo_y = round(x), round(y)  # 将 x 和 y 坐标四舍五入并赋值给 torpedo_x 和 torpedo_y
            if torpedo_x < 0 or torpedo_x > 7 or torpedo_y < 0 or torpedo_y > 7:  # 如果 torpedo_x 或 torpedo_y 超出了范围
                print("TORPEDO MISSED")  # 打印“TORPEDO MISSED”
                self.klingons_fire()  # 调用 klingons_fire 方法
                return  # 退出函数
            print(f"                {torpedo_x + 1} , {torpedo_y + 1}")  # 打印 torpedo_x 和 torpedo_y 的值
            if world.quadrant.get_value(torpedo_x, torpedo_y) != Entity.void:  # 如果指定坐标处的值不是空白
                break  # 跳出循环

        if world.quadrant.get_value(torpedo_x, torpedo_y) == Entity.klingon:  # 如果指定坐标处的值是克林贡飞船
            print("*** KLINGON DESTROYED ***")  # 打印“*** KLINGON DESTROYED ***”
            self.world.quadrant.nb_klingons -= 1  # 更新克林贡飞船数量
            world.remaining_klingons -= 1  # 更新剩余克林贡飞船数量
            if world.remaining_klingons <= 0:  # 如果剩余克林贡飞船数量小于等于 0
                self.end_game(won=True, quit=False)  # 调用 end_game 方法，传入参数 won=True, quit=False
                return  # 退出函数
            for klingon_ship in klingon_ships:  # 遍历克林贡飞船列表
# 如果鱼雷的 x 坐标等于克林贡飞船的区块 x 坐标，并且鱼雷的 y 坐标等于克林贡飞船的区块 y 坐标
if (
    torpedo_x == klingon_ship.sector.x
    and torpedo_y == klingon_ship.sector.y
):
    # 设置克林贡飞船的护盾为 0
    klingon_ship.shield = 0
# 如果鱼雷的 x、y 坐标对应的象限值为星体
elif world.quadrant.get_value(torpedo_x, torpedo_y) == Entity.star:
    # 打印鱼雷能量被星体吸收的消息
    print(f"STAR AT {torpedo_x + 1} , {torpedo_y + 1} ABSORBED TORPEDO ENERGY.")
    # 克林贡人发射鱼雷
    self.klingons_fire()
    # 返回
    return
# 如果鱼雷的 x、y 坐标对应的象限值为星舰基地
elif world.quadrant.get_value(torpedo_x, torpedo_y) == Entity.starbase:
    # 打印星舰基地被摧毁的消息
    print("*** STARBASE DESTROYED ***")
    # 世界象限中的星舰基地数量减一
    self.world.quadrant.nb_bases -= 1
    # 星系中的星舰基地数量减一
    world.bases_in_galaxy -= 1
    # 如果星系中的星舰基地数量为 0，并且剩余的克林贡人数量小于等于当前星期数减去初始星期数减去任务持续时间
    if (
        world.bases_in_galaxy == 0
        and world.remaining_klingons
        <= world.stardate - world.initial_stardate - world.mission_duration
    ):
        # 打印指挥官被免除指挥并被判处在 Cygnus 12 上进行 99 个星期的苦役的消息
        print("THAT DOES IT, CAPTAIN!! YOU ARE HEREBY RELIEVED OF COMMAND")
        print("AND SENTENCED TO 99 STARDATES AT HARD LABOR ON CYGNUS 12!!")
                # 调用 end_game 方法，传入参数 won=False
                self.end_game(won=False)
                # 返回
                return
            # 打印信息
            print("STARFLEET COMMAND REVIEWING YOUR RECORD TO CONSIDER")
            print("COURT MARTIAL!")
            # 将飞船的 docked 属性设置为 False

        # 在世界的象限中设置指定位置的值为 Entity.void
        world.quadrant.set_value(torpedo_x, torpedo_y, Entity.void)
        # 在世界的星系地图中更新飞船所在象限的数据
        world.galaxy_map[ship.position.quadrant.x][
            ship.position.quadrant.y
        ] = QuadrantData(
            self.world.quadrant.nb_klingons,
            self.world.quadrant.nb_bases,
            self.world.quadrant.nb_stars,
        )
        # 在已绘制的星系地图中更新飞船所在象限的数据
        world.charted_galaxy_map[ship.position.quadrant.x][
            ship.position.quadrant.y
        ] = world.galaxy_map[ship.position.quadrant.x][ship.position.quadrant.y]
        # 调用 klingons_fire 方法
        self.klingons_fire()

    # 定义 short_range_scan 方法，返回类型为 None
    def short_range_scan(self) -> None:
        """Print a short range scan."""  # 打印一个短程扫描
        self.world.ship.docked = False  # 将飞船的停靠状态设置为False
        ship = self.world.ship  # 将self.world.ship赋值给ship变量
        for x in (  # 遍历x坐标的范围
            ship.position.sector.x - 1,  # 飞船所在扇区的x坐标减1
            ship.position.sector.x,  # 飞船所在扇区的x坐标
            ship.position.sector.x + 1,  # 飞船所在扇区的x坐标加1
        ):
            for y in (  # 遍历y坐标的范围
                ship.position.sector.y - 1,  # 飞船所在扇区的y坐标减1
                ship.position.sector.y,  # 飞船所在扇区的y坐标
                ship.position.sector.y + 1,  # 飞船所在扇区的y坐标加1
            ):
                if (  # 如果以下条件成立
                    0 <= x <= 7  # x坐标在0到7之间
                    and 0 <= y <= 7  # y坐标在0到7之间
                    and self.world.quadrant.get_value(x, y) == Entity.starbase  # self.world.quadrant在(x, y)位置的值为Entity.starbase
                ):
                    ship.docked = True  # 将飞船的停靠状态设置为True
                    cs = "DOCKED"  # 将cs变量设置为"DOCKED"
                    ship.refill()  # 重新填充飞船的能量
                    print("SHIELDS DROPPED FOR DOCKING PURPOSES")  # 打印信息，表示为对接目的已关闭护盾
                    ship.shields = 0  # 将飞船的护盾能量设置为0
                    break  # 跳出当前循环
            else:  # 如果上述条件不满足
                continue  # 继续下一次循环
            break  # 跳出当前循环
        else:  # 如果上述条件不满足
            if self.world.quadrant.nb_klingons > 0:  # 如果当前象限内克林贡星舰数量大于0
                cs = "*RED*"  # 将cs设置为"*RED*"
            elif ship.energy < Ship.energy_capacity * 0.1:  # 如果飞船能量小于能量容量的10%
                cs = "YELLOW"  # 将cs设置为"YELLOW"
            else:  # 如果上述条件都不满足
                cs = "GREEN"  # 将cs设置为"GREEN"

        if ship.damage_stats[1] < 0:  # 如果飞船的损坏状态中第二个元素小于0
            print("\n*** SHORT RANGE SENSORS ARE OUT ***\n")  # 打印信息，表示短程传感器已损坏
            return  # 返回

        sep = "---------------------------------"  # 将sep设置为"---------------------------------"
        # 打印分隔线
        print(sep)
        # 遍历行
        for x in range(8):
            line = ""
            # 遍历列
            for y in range(8):
                # 将当前象限的值添加到行中
                line = line + " " + self.world.quadrant.data[x][y].value

            # 根据行数添加相应的信息到行末
            if x == 0:
                line += f"        STARDATE           {round(int(self.world.stardate * 10) * 0.1, 1)}"
            elif x == 1:
                line += f"        CONDITION          {cs}"
            elif x == 2:
                line += f"        QUADRANT           {ship.position.quadrant}"
            elif x == 3:
                line += f"        SECTOR             {ship.position.sector}"
            elif x == 4:
                line += f"        PHOTON TORPEDOES   {int(ship.torpedoes)}"
            elif x == 5:
                line += f"        TOTAL ENERGY       {int(ship.energy + ship.shields)}"
            elif x == 6:
                line += f"        SHIELDS            {int(ship.shields)}"
            else:
                # 如果不是星球，则打印剩余克林贡人数
                line += f"        KLINGONS REMAINING {self.world.remaining_klingons}"

            # 打印行
            print(line)
        # 打印分隔线
        print(sep)

    def long_range_scan(self) -> None:
        """Print a long range scan."""
        # 如果飞船的第三项损伤统计小于0，则打印长程传感器不可用并返回
        if self.world.ship.damage_stats[2] < 0:
            print("LONG RANGE SENSORS ARE INOPERABLE")
            return

        # 打印当前象限的长程扫描结果
        print(f"LONG RANGE SCAN FOR QUADRANT {self.world.ship.position.quadrant}")
        print_scan_results(
            self.world.ship.position.quadrant,
            self.world.galaxy_map,
            self.world.charted_galaxy_map,
        )

    def navigation(self) -> None:
        """
        Take navigation input and move the Enterprise.

        1/8 warp goes 1 sector in the direction dirs[course]
        """
        # 获取世界和飞船对象
        world = self.world
        ship = world.ship

        # 获取用户输入的航向，并将其转换为0-8的范围
        cd = get_user_float("COURSE (1-9)? ") - 1  # Convert to 0-8
        # 如果cd等于dirs列表的长度减1，则将cd设为0
        if cd == len(dirs) - 1:
            cd = 0
        # 如果cd小于0或大于等于dirs列表的长度，则打印错误信息并返回
        if cd < 0 or cd >= len(dirs):
            print("   LT. SULU REPORTS, 'INCORRECT COURSE DATA, SIR!'")
            return

        # 获取用户输入的warp因子，并根据飞船的损伤状态调整最大值
        warp = get_user_float(
            f"WARP FACTOR (0-{'0.2' if ship.damage_stats[0] < 0 else '8'})? "
        )
        # 如果飞船的损伤状态小于0且warp大于0.2，则打印错误信息
        if ship.damage_stats[0] < 0 and warp > 0.2:
            print("WARP ENGINES ARE DAMAGED. MAXIMUM SPEED = WARP 0.2")
        return  # 如果条件不满足，直接返回，结束函数执行
        if warp == 0:  # 如果warp为0，直接返回，结束函数执行
            return
        if warp < 0 or warp > 8:  # 如果warp小于0或者大于8，打印报错信息并返回，结束函数执行
            print(
                f"   CHIEF ENGINEER SCOTT REPORTS 'THE ENGINES WON'T TAKE WARP {warp}!'"
            )
            return

        warp_rounds = round(warp * 8)  # 计算warp乘以8的结果并四舍五入
        if ship.energy < warp_rounds:  # 如果飞船能量小于warp_rounds，打印报错信息并返回，结束函数执行
            print("ENGINEERING REPORTS   'INSUFFICIENT ENERGY AVAILABLE")
            print(f"                       FOR MANEUVERING AT WARP {warp}!'")
            if ship.shields >= warp_rounds - ship.energy and ship.damage_stats[6] >= 0:  # 如果飞船护盾大于等于warp_rounds减去飞船能量，并且飞船损伤状态的第七个元素大于等于0，打印相关信息并返回，结束函数执行
                print(
                    f"DEFLECTOR CONTROL ROOM ACKNOWLEDGES {ship.shields} UNITS OF ENERGY"
                )
                print("                         PRESENTLY DEPLOYED TO SHIELDS.")
            return
        # klingons move and fire
        # Klingon船只移动并开火
        for klingon_ship in self.world.quadrant.klingon_ships:
            # 如果 Klingon船只的护盾不为0
            if klingon_ship.shield != 0:
                # 在世界象限中设置指定位置的值为空
                world.quadrant.set_value(
                    klingon_ship.sector.x, klingon_ship.sector.y, Entity.void
                )
                # 找到一个空位置并将 Klingon船只移动到该位置
                (
                    klingon_ship.sector.x,
                    klingon_ship.sector.y,
                ) = world.quadrant.find_empty_place()
                # 在世界象限中设置指定位置的值为 Klingon
                world.quadrant.set_value(
                    klingon_ship.sector.x, klingon_ship.sector.y, Entity.klingon
                )

        # 让 Klingon船只开火
        self.klingons_fire()

        # repair damaged devices and print damage report
        # 修复受损设备并打印损坏报告
        line = ""
        # 遍历8次
        for i in range(8):
            # 如果飞船的损坏统计值小于0
            if ship.damage_stats[i] < 0:
                # 更新飞船的损坏统计信息，根据warp和1的最小值来增加相应设备的损坏值
                ship.damage_stats[i] += min(warp, 1)
                # 如果设备的损坏值在-0.1到0之间，则将其设置为-0.1
                if -0.1 < ship.damage_stats[i] < 0:
                    ship.damage_stats[i] = -0.1
                # 如果设备的损坏值大于等于0
                elif ship.damage_stats[i] >= 0:
                    # 如果line为空，则将其设置为"DAMAGE CONTROL REPORT:"
                    if len(line) == 0:
                        line = "DAMAGE CONTROL REPORT:"
                    # 在line后面添加设备修复完成的信息
                    line += f"   {ship.devices[i]} REPAIR COMPLETED\n"
        # 如果line的长度大于0，则打印line
        if len(line) > 0:
            print(line)
        # 如果随机数小于等于0.2
        if random.random() <= 0.2:
            # 随机选择一个设备
            device = fnr()
            # 如果随机数小于0.6
            if random.random() < 0.6:
                # 减少相应设备的损坏值
                ship.damage_stats[device] -= random.random() * 5 + 1
                # 打印设备损坏的信息
                print(f"DAMAGE CONTROL REPORT:   {ship.devices[device]} DAMAGED\n")
            # 如果随机数不小于0.6
            else:
                # 增加相应设备的损坏值
                ship.damage_stats[device] += random.random() * 3 + 1
                # 打印设备修复的信息
                print(
                    f"DAMAGE CONTROL REPORT:   {ship.devices[device]} STATE OF REPAIR IMPROVED\n"
                )
        self.move_ship(warp_rounds, cd)  # 调用move_ship方法，移动飞船
        world.stardate += 0.1 * int(10 * warp) if warp < 1 else 1  # 更新星际日期
        if world.has_mission_ended():  # 检查任务是否结束
            self.end_game(won=False, quit=False)  # 结束游戏
            return

        self.short_range_scan()  # 进行短程扫描

    def move_ship(self, warp_rounds: int, cd: float) -> None:  # 定义移动飞船的方法，参数为warp_rounds和cd

        assert cd >= 0  # 断言cd大于等于0
        assert cd < len(dirs) - 1  # 断言cd小于dirs的长度减1
        # cd is the course data which points to 'dirs'  # cd是指向'dirs'的航向数据
        world = self.world  # 获取世界对象
        ship = self.world.ship  # 获取世界中的飞船对象
        world.quadrant.set_value(  # 设置世界象限的值
            int(ship.position.sector.x), int(ship.position.sector.y), Entity.void
        )
        cdi = int(cd)  # 将cd转换为整数

        # Interpolate direction:  # 插值方向
        dx = dirs[cdi][0] + (dirs[cdi + 1][0] - dirs[cdi][0]) * (cd - cdi)  # 计算x方向的增量
        dy = dirs[cdi][1] + (dirs[cdi + 1][1] - dirs[cdi][1]) * (cd - cdi)  # 计算y方向的增量

        start_quadrant = Point(ship.position.quadrant.x, ship.position.quadrant.y)  # 保存起始象限坐标
        sector_start_x: float = ship.position.sector.x  # 保存起始扇区x坐标
        sector_start_y: float = ship.position.sector.y  # 保存起始扇区y坐标

        for _ in range(warp_rounds):  # 循环warp_rounds次
            ship.position.sector.x += dx  # type: ignore  # 更新飞船在扇区中的x坐标
            ship.position.sector.y += dy  # type: ignore  # 更新飞船在扇区中的y坐标

            if (
                ship.position.sector.x < 0  # 如果飞船在x轴上超出了扇区范围
                or ship.position.sector.x > 7  # 如果飞船在x轴上超出了扇区范围
                or ship.position.sector.y < 0  # 如果飞船在y轴上超出了扇区范围
                or ship.position.sector.y > 7  # 如果飞船在y轴上超出了扇区范围
            ):
                # exceeded quadrant limits; calculate final position
                sector_start_x += ship.position.quadrant.x * 8 + warp_rounds * dx  # 计算最终x坐标
                sector_start_y += ship.position.quadrant.y * 8 + warp_rounds * dy  # 计算最终y坐标
# 将飞船的位置信息转换为象限和扇区坐标
ship.position.quadrant.x = int(sector_start_x / 8)  # 计算飞船所在象限的 x 坐标
ship.position.quadrant.y = int(sector_start_y / 8)  # 计算飞船所在象限的 y 坐标
ship.position.sector.x = int(sector_start_x - ship.position.quadrant.x * 8)  # 计算飞船所在扇区的 x 坐标
ship.position.sector.y = int(sector_start_y - ship.position.quadrant.y * 8)  # 计算飞船所在扇区的 y 坐标

# 处理飞船位置超出边界的情况
if ship.position.sector.x < 0:  # 如果飞船所在扇区的 x 坐标小于 0
    ship.position.quadrant.x -= 1  # 飞船所在象限的 x 坐标减 1
    ship.position.sector.x = 7  # 飞船所在扇区的 x 坐标设为 7
if ship.position.sector.y < 0:  # 如果飞船所在扇区的 y 坐标小于 0
    ship.position.quadrant.y -= 1  # 飞船所在象限的 y 坐标减 1
    ship.position.sector.y = 7  # 飞船所在扇区的 y 坐标设为 7

# 处理飞船位置超出边界的情况
hit_edge = False  # 初始化是否碰到边界的标志为 False
if ship.position.quadrant.x < 0:  # 如果飞船所在象限的 x 坐标小于 0
    hit_edge = True  # 设置碰到边界的标志为 True
    ship.position.quadrant.x = ship.position.sector.x = 0  # 将飞船所在象限和扇区的 x 坐标设为 0
if ship.position.quadrant.x > 7:  # 如果飞船所在象限的 x 坐标大于 7
                    hit_edge = True  # 设置标志位，表示飞船到达了边缘
                    ship.position.quadrant.x = ship.position.sector.x = 7  # 将飞船的 x 坐标设置为 7
                if ship.position.quadrant.y < 0:  # 如果飞船的 y 坐标小于 0
                    hit_edge = True  # 设置标志位，表示飞船到达了边缘
                    ship.position.quadrant.y = ship.position.sector.y = 0  # 将飞船的 y 坐标设置为 0
                if ship.position.quadrant.y > 7:  # 如果飞船的 y 坐标大于 7
                    hit_edge = True  # 设置标志位，表示飞船到达了边缘
                    ship.position.quadrant.y = ship.position.sector.y = 7  # 将飞船的 y 坐标设置为 7
                if hit_edge:  # 如果飞船到达了边缘
                    print("LT. UHURA REPORTS MESSAGE FROM STARFLEET COMMAND:")  # 打印消息
                    print("  'PERMISSION TO ATTEMPT CROSSING OF GALACTIC PERIMETER")  # 打印消息
                    print("  IS HEREBY *DENIED*. SHUT DOWN YOUR ENGINES.'")  # 打印消息
                    print("CHIEF ENGINEER SCOTT REPORTS  'WARP ENGINES SHUT DOWN")  # 打印消息
                    print(
                        f"  AT SECTOR {ship.position.sector} OF "
                        f"QUADRANT {ship.position.quadrant}.'"
                    )  # 打印消息，包含飞船的位置信息
                    if world.has_mission_ended():  # 如果任务已经结束
                        self.end_game(won=False, quit=False)  # 结束游戏，飞船失败，不是退出游戏
                        return  # 返回
                # 检查飞船是否停留在指定的象限
                stayed_in_quadrant = (
                    ship.position.quadrant.x == start_quadrant.x
                    and ship.position.quadrant.y == start_quadrant.y
                )
                # 如果飞船停留在指定的象限，则跳出循环
                if stayed_in_quadrant:
                    break
                # 更新星际日期
                world.stardate += 1
                # 调用飞船的机动能量方法
                ship.maneuver_energy(warp_rounds)
                # 调用新象限的方法
                self.new_quadrant()
                # 返回
                return
            # 获取飞船所在的扇区
            ship_sector = self.world.ship.position.sector
            # 获取飞船所在扇区的 x 坐标
            ship_x = int(ship_sector.x)
            # 获取飞船所在扇区的 y 坐标
            ship_y = int(ship_sector.y)
            # 如果飞船所在象限的数据不是空的实体
            if self.world.quadrant.data[ship_x][ship_y] != Entity.void:
                # 调整飞船所在扇区的 x 和 y 坐标
                ship_sector.x = int(ship_sector.x - dx)
                ship_sector.y = int(ship_sector.y - dy)
                # 打印错误信息
                print(
                    "WARP ENGINES SHUT DOWN AT SECTOR "
                    f"{ship_sector} DUE TO BAD NAVIGATION"
                )
                break  # 结束循环
        else:  # 如果没有break，则执行以下代码
            ship.position.sector.x, ship.position.sector.y = int(
                ship.position.sector.x
            ), int(ship.position.sector.y)  # 将ship.position.sector.x和ship.position.sector.y转换为整数类型

        world.quadrant.set_value(
            int(ship.position.sector.x), int(ship.position.sector.y), Entity.ship
        )  # 设置world.quadrant中ship的位置坐标
        ship.maneuver_energy(warp_rounds)  # 调用ship的maneuver_energy方法，传入warp_rounds参数

    def damage_control(self) -> None:  # 定义damage_control方法，返回类型为None
        """Print a damage control report."""  # 打印一个损坏控制报告
        ship = self.world.ship  # 获取self.world中的ship对象

        if ship.damage_stats[5] < 0:  # 如果ship的damage_stats中第5个元素小于0
            print("DAMAGE CONTROL REPORT NOT AVAILABLE")  # 打印"DAMAGE CONTROL REPORT NOT AVAILABLE"
        else:  # 否则
            print("\nDEVICE             STATE OF REPAIR")  # 打印"\nDEVICE             STATE OF REPAIR"
# 遍历范围为8的循环，打印每个设备的名称和损坏程度
for r1 in range(8):
    print(
        f"{ship.devices[r1].ljust(26, ' ')}{int(ship.damage_stats[r1] * 100) * 0.01:g}"
    )
print()

# 如果飞船没有停靠，则返回
if not ship.docked:
    return

# 计算损坏程度小于0的设备数量，并将其总和赋给damage_sum
damage_sum = sum(0.1 for i in range(8) if ship.damage_stats[i] < 0)
# 如果没有损坏的设备，则返回
if damage_sum == 0:
    return

# 将损坏设备的修复时间加上基地修复延迟时间
damage_sum += self.world.quadrant.delay_in_repairs_at_base
# 如果修复时间大于等于1，则将其设为0.9
if damage_sum >= 1:
    damage_sum = 0.9
# 打印修复信息
print("\nTECHNICIANS STANDING BY TO EFFECT REPAIRS TO YOUR SHIP;")
print(
    f"ESTIMATED TIME TO REPAIR: {round(0.01 * int(100 * damage_sum), 2)} STARDATES"
)
        # 如果用户输入不是Y，则返回
        if input("WILL YOU AUTHORIZE THE REPAIR ORDER (Y/N)? ").upper().strip() != "Y":
            return

        # 循环遍历船只的损坏统计数据，将小于0的值设为0
        for i in range(8):
            if ship.damage_stats[i] < 0:
                ship.damage_stats[i] = 0
        # 更新星际日期
        self.world.stardate += damage_sum + 0.1

    def computer(self) -> None:
        """执行图书馆计算机的各种功能。"""
        world = self.world
        ship = world.ship

        # 如果船只的第7项损坏统计数据小于0，则打印"COMPUTER DISABLED"并返回
        if ship.damage_stats[7] < 0:
            print("COMPUTER DISABLED")
            return

        # 无限循环，等待用户输入命令
        while True:
            command = input("COMPUTER ACTIVE AND AWAITING COMMAND? ")
            # 如果用户输入为空，则继续等待命令
            if len(command) == 0:
                com = 6  # 设置默认的命令值为6
            else:  # 如果命令值不为空
                try:  # 尝试将命令值转换为整数
                    com = int(command)
                except ValueError:  # 如果转换失败，则将命令值设置为6
                    com = 6
            if com < 0:  # 如果命令值小于0，则返回
                return

            print()  # 打印空行

            if com in [0, 5]:  # 如果命令值为0或5
                if com == 5:  # 如果命令值为5
                    print("                        THE GALAXY")  # 打印"THE GALAXY"
                else:  # 如果命令值为0
                    print(
                        "\n        COMPUTER RECORD OF GALAXY FOR "
                        f"QUADRANT {ship.position.quadrant}\n"
                    )  # 打印"COMPUTER RECORD OF GALAXY FOR QUADRANT"后跟飞船所在象限的值
                # 打印表头
                print("       1     2     3     4     5     6     7     8")
                # 打印分隔线
                sep = "     ----- ----- ----- ----- ----- ----- ----- -----"
                print(sep)

                # 遍历每一行
                for i in range(8):
                    line = " " + str(i + 1) + " "

                    # 如果com等于5，则执行以下操作
                    if com == 5:
                        # 获取象限名称并添加到行中
                        g2s = Quadrant.quadrant_name(i, 0, True)
                        line += (" " * int(12 - 0.5 * len(g2s))) + g2s
                        g2s = Quadrant.quadrant_name(i, 4, True)
                        line += (" " * int(39 - 0.5 * len(g2s) - len(line))) + g2s
                    # 如果com不等于5，则执行以下操作
                    else:
                        # 遍历每一列
                        for j in range(8):
                            line += "   "
                            # 如果星系地图中该位置的数字为0，则添加星号到行中
                            if world.charted_galaxy_map[i][j].num() == 0:
                                line += "***"
                            # 否则，将星系地图中该位置的数字加上1000后添加到行中
                            else:
                                line += str(world.charted_galaxy_map[i][j].num() + 1000)
# 如果 com 的值为 0，则执行以下代码块
if com == 0:
    # 从 line 中提取最后三个字符
    sep = line[-3:]
    # 打印 line 变量的值
    print(line)
    # 打印 sep 变量的值
    print(sep)
    # 打印空行
    print()
# 如果 com 的值为 1，则执行以下代码块
elif com == 1:
    # 打印 "   STATUS REPORT:"
    print("   STATUS REPORT:")
    # 打印剩余克林贡人数
    print(f"KLINGON{'S' if world.remaining_klingons > 1 else ''} LEFT: {world.remaining_klingons}")
    # 打印任务剩余时间
    print("MISSION MUST BE COMPLETED IN " f"{round(0.1 * int(world.remaining_time() * 10), 1)} STARDATES")

    # 如果星球上没有星舰基地
    if world.bases_in_galaxy == 0:
        # 打印提示信息
        print("YOUR STUPIDITY HAS LEFT YOU ON YOUR OWN IN")
        print("  THE GALAXY -- YOU HAVE NO STARBASES LEFT!")
    else:
                    print(
                        f"THE FEDERATION IS MAINTAINING {world.bases_in_galaxy} "
                        f"STARBASE{'S' if world.bases_in_galaxy > 1 else ''} IN THE GALAXY"
                    )
```
这段代码是在打印输出一个关于星际基地数量的信息，根据星际基地的数量来决定输出单数还是复数形式。

```
                self.damage_control()
```
这段代码调用了一个名为damage_control的方法，用于处理飞船的损坏情况。

```
            elif com == 2:
                if self.world.quadrant.nb_klingons <= 0:
                    print(
                        "SCIENCE OFFICER SPOCK REPORTS  'SENSORS SHOW NO ENEMY "
                        "SHIPS\n"
                        "                                IN THIS QUADRANT'"
                    )
                    return
```
这段代码是在检查当前象限是否有克林贡战舰，如果没有则打印一条相应的信息并返回。

```
                print(
                    f"FROM ENTERPRISE TO KLINGON BATTLE CRUISER{'S' if self.world.quadrant.nb_klingons > 1 else ''}"
                )
```
这段代码是在打印输出一条信息，根据当前象限内克林贡战舰的数量来决定输出单数还是复数形式。

```
                for klingon_ship in self.world.quadrant.klingon_ships:
```
这段代码是在遍历当前象限内的克林贡战舰列表。
                    # 如果克林贡飞船的护盾大于0，则打印方向
                    if klingon_ship.shield > 0:
                        print_direction(
                            Point(ship.position.sector.x, ship.position.sector.y),  # 打印飞船当前位置
                            Point(
                                int(klingon_ship.sector.x),  # 打印克林贡飞船的x坐标
                                int(klingon_ship.sector.y),  # 打印克林贡飞船的y坐标
                            ),
                        )
            elif com == 3:  # 如果命令是3
                if self.world.quadrant.nb_bases == 0:  # 如果当前象限没有星舰基地
                    print(
                        "MR. SPOCK REPORTS,  'SENSORS SHOW NO STARBASES IN THIS "
                        "QUADRANT.'"
                    )  # 打印无星舰基地的信息
                    return  # 返回
                # 打印从企业到星舰基地的方向
                print("FROM ENTERPRISE TO STARBASE:")
                print_direction(
                    Point(ship.position.sector.x, ship.position.sector.y),  # 打印飞船当前位置
                    self.world.quadrant.starbase,  # 打印星舰基地的位置
                )
            elif com == 4:  # 如果用户输入的命令是4
                print("DIRECTION/DISTANCE CALCULATOR:")  # 打印提示信息
                print(
                    f"YOU ARE AT QUADRANT {ship.position.quadrant} "  # 打印飞船所在的象限
                    f"SECTOR {ship.position.sector}"  # 打印飞船所在的扇区
                )
                print("PLEASE ENTER")  # 打印提示信息
                while True:  # 进入无限循环
                    coordinates = input("  INITIAL COORDINATES (X,Y)? ").split(",")  # 获取用户输入的初始坐标
                    if len(coordinates) == 2:  # 如果用户输入的坐标是两个
                        from1, from2 = int(coordinates[0]) - 1, int(coordinates[1]) - 1  # 将用户输入的坐标转换为整数并减去1
                        if 0 <= from1 <= 7 and 0 <= from2 <= 7:  # 如果坐标在合法范围内
                            break  # 退出循环
                while True:  # 进入无限循环
                    coordinates = input("  FINAL COORDINATES (X,Y)? ").split(",")  # 获取用户输入的最终坐标
                    if len(coordinates) == 2:  # 如果用户输入的坐标是两个
                        to1, to2 = int(coordinates[0]) - 1, int(coordinates[1]) - 1  # 将用户输入的坐标转换为整数并减去1
                        if 0 <= to1 <= 7 and 0 <= to2 <= 7:  # 如果坐标在合法范围内
                            break  # 退出循环
                print_direction(Point(from1, from2), Point(to1, to2))  # 调用print_direction函数打印从(from1, from2)到(to1, to2)的方向
            else:
                print(  # 打印以下信息
                    "FUNCTIONS AVAILABLE FROM LIBRARY-COMPUTER:\n"
                    "   0 = CUMULATIVE GALACTIC RECORD\n"
                    "   1 = STATUS REPORT\n"
                    "   2 = PHOTON TORPEDO DATA\n"
                    "   3 = STARBASE NAV DATA\n"
                    "   4 = DIRECTION/DISTANCE CALCULATOR\n"
                    "   5 = GALAXY 'REGION NAME' MAP\n"
                )

    def end_game(
        self, won: bool = False, quit: bool = True, enterprise_killed: bool = False
    ) -> None:
        """Handle end-of-game situations."""
        if won:  # 如果游戏胜利
            print("CONGRATULATIONS, CAPTAIN! THE LAST KLINGON BATTLE CRUISER")  # 打印恭喜信息
            print("MENACING THE FEDERATION HAS BEEN DESTROYED.\n")  # 打印信息
            print(  # 打印以下信息
                f"YOUR EFFICIENCY RATING IS {round(1000 * (self.world.remaining_klingons / (self.world.stardate - self.world.initial_stardate))**2, 4)}\n\n"
            )
```
这行代码是一个字符串格式化输出，根据游戏中的一些参数计算玩家的效率评级并将其打印出来。

```python
        else:
            if not quit:
                if enterprise_killed:
                    print(
                        "\nTHE ENTERPRISE HAS BEEN DESTROYED. THE FEDERATION "
                        "WILL BE CONQUERED."
                    )
                print(f"IT IS STARDATE {round(self.world.stardate, 1)}")
```
这部分代码包含了一些条件语句和字符串格式化输出，根据游戏中的一些参数打印不同的消息。

```python
            print(
                f"THERE WERE {self.world.remaining_klingons} KLINGON BATTLE CRUISERS LEFT AT"
            )
            print("THE END OF YOUR MISSION.\n\n")
```
这部分代码也是字符串格式化输出，打印游戏中剩余的 Klingon 战舰数量。

```python
            if self.world.bases_in_galaxy == 0:
                sys.exit()
```
这是一个条件语句，如果星际基地数量为0，则退出游戏。

```python
        print("THE FEDERATION IS IN NEED OF A NEW STARSHIP COMMANDER")
```
这行代码打印出一条消息，表示联邦需要一位新的星际舰队指挥官。
        print("FOR A SIMILAR MISSION -- IF THERE IS A VOLUNTEER,")  # 打印提示信息
        if input("LET HIM STEP FORWARD AND ENTER 'AYE'? ").upper().strip() != "AYE":  # 获取用户输入，如果不是'AYE'，则退出程序
            sys.exit()
        self.restart = True  # 设置self.restart为True，表示重新开始

klingon_shield_strength: Final = 200  # 定义一个名为klingon_shield_strength的常量，值为200
dirs: Final = [  # 定义一个名为dirs的常量列表
    [0, 1],  # 1: 向右移动 (同#9)
    [-1, 1],  # 2: 向右上移动
    [-1, 0],  # 3: 向上移动（x坐标减小；北方）
    [-1, -1],  # 4: 向左上移动（西北方）
    [0, -1],  # 5: 向左移动（西方）
    [1, -1],  # 6: 向左下移动（西南方）
    [1, 0],  # 7: 向下移动（x坐标增大；南方）
    [1, 1],  # 8: 向右下移动
    [0, 1],  # 9: 向右移动（东方）
]  # 各个方向的向量

def fnr() -> int:
    """Generate a random integer from 0 to 7 inclusive."""
    # 定义一个函数，返回一个随机整数，范围在0到7之间（包括0和7）
    return random.randint(0, 7)


def print_scan_results(
    quadrant: Point,
    galaxy_map: List[List[QuadrantData]],
    charted_galaxy_map: List[List[QuadrantData]],
) -> None:
    sep = "-------------------"
    # 打印分隔线
    print(sep)
    for x in (quadrant.x - 1, quadrant.x, quadrant.x + 1):
        n: List[Optional[int]] = [None, None, None]

        # Reveal parts of the current map
        # 揭示当前地图的部分内容
        for y in (quadrant.y - 1, quadrant.y, quadrant.y + 1):
            if 0 <= x <= 7 and 0 <= y <= 7:
                # 如果x和y的值在0到7之间，获取galaxy_map中对应位置的数据
                n[y - quadrant.y + 1] = galaxy_map[x][y].num()
                charted_galaxy_map[x][y] = galaxy_map[x][y]  # 将星系地图中的特定位置的值复制到另一个地图中

        line = ": "  # 初始化一个字符串变量
        for line_col in n:  # 遍历n中的元素
            if line_col is None:  # 如果元素为None
                line += "*** : "  # 在字符串变量中添加特定格式的字符串
            else:
                line += str(line_col + 1000).rjust(4, " ")[-3:] + " : "  # 在字符串变量中添加特定格式的字符串
        print(line)  # 打印字符串变量
        print(sep)  # 打印分隔符

def print_direction(source: Point, to: Point) -> None:
    """Print direction and distance between two locations in the grid."""
    delta1 = -(to.x - source.x)  # 计算两个位置在x轴上的距离，并取负值
    delta2 = to.y - source.y  # 计算两个位置在y轴上的距离

    if delta2 > 0:  # 如果位置2在位置1的上方
        if delta1 < 0:  # 如果位置2在位置1的左侧
            base = 7  # 设置基准值为7
    else:  # 如果delta1小于等于0
        if delta1 > 0:  # 如果delta1大于0
            base = 3  # 将base设置为3
        else:  # 否则
            base = 5  # 将base设置为5
            delta1, delta2 = delta2, delta1  # 交换delta1和delta2的值

    delta1, delta2 = abs(delta1), abs(delta2)  # 将delta1和delta2取绝对值

    if delta1 > 0 or delta2 > 0:  # 如果delta1大于0或者delta2大于0（原始代码中的bug；没有检查除以0的情况）
        if delta1 >= delta2:  # 如果delta1大于等于delta2
            print(f"DIRECTION = {round(base + delta2 / delta1, 6)}")  # 打印方向
        else:  # 否则
            print(f"DIRECTION = {round(base + 2 - delta1 / delta2, 6)}")  # 打印方向

    print(f"DISTANCE = {round(sqrt(delta1 ** 2 + delta2 ** 2), 6)}")  # 打印距离
def main() -> None:
    # 创建游戏对象
    game = Game()
    # 获取游戏世界对象
    world = game.world
    # 获取世界中的飞船对象
    ship = world.ship

    # 创建命令字典，将命令字符串映射到游戏对象的对应方法
    f: Dict[str, Callable[[], None]] = {
        "NAV": game.navigation,  # 导航
        "SRS": game.short_range_scan,  # 短程扫描
        "LRS": game.long_range_scan,  # 长程扫描
        "PHA": game.phaser_control,  # 相位炮控制
        "TOR": game.photon_torpedoes,  # 光子鱼雷
        "SHE": ship.shield_control,  # 护盾控制
        "DAM": game.damage_control,  # 损坏控制
        "COM": game.computer,  # 计算机
        "XXX": game.end_game,  # 结束游戏
    }

    # 循环执行游戏的启动方法
    while True:
        game.startup()
        # 调用游戏对象的 new_quadrant 方法，生成新的游戏象限
        game.new_quadrant()
        # 将 restart 变量设置为 False
        restart = False

        # 当 restart 为 False 时执行循环
        while not restart:
            # 如果飞船的护盾加上能量小于等于10，或者能量小于等于10且损坏状态不为0
            if ship.shields + ship.energy <= 10 or (
                ship.energy <= 10 and ship.damage_stats[6] != 0
            ):
                # 打印错误信息
                print(
                    "\n** FATAL ERROR **   YOU'VE JUST STRANDED YOUR SHIP "
                    "IN SPACE.\nYOU HAVE INSUFFICIENT MANEUVERING ENERGY, "
                    "AND SHIELD CONTROL\nIS PRESENTLY INCAPABLE OF CROSS-"
                    "CIRCUITING TO ENGINE ROOM!!"
                )

            # 获取用户输入的命令
            command = input("COMMAND? ").upper().strip()

            # 如果输入的命令在 f 中
            if command in f:
                # 调用 f 中对应命令的方法
                f[command]()
            else:
                # 打印错误信息
                print(
# 打印提示信息，要求用户输入以下选项之一
# NAV  (设置航向)
# SRS  (进行短程传感器扫描)
# LRS  (进行长程传感器扫描)
# PHA  (开火激光炮)
# TOR  (发射光子鱼雷)
# SHE  (升降护盾)
# DAM  (报告损坏情况)
# COM  (调用图书馆计算机)
# XXX  (辞去指挥权)
# 用户输入后，程序将根据用户选择执行相应的操作

if __name__ == "__main__":
    main()
# 如果当前脚本被直接执行，则调用main()函数，开始执行程序
```