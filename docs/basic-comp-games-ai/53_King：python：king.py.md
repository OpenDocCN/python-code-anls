# `53_King\python\king.py`

```
"""
KING

A strategy game where the player is the king.

Ported to Python by Martin Thoma in 2022
"""

import sys  # 导入 sys 模块，用于访问与 Python 解释器交互的变量和函数
from dataclasses import dataclass  # 从 dataclasses 模块中导入 dataclass 类装饰器，用于创建数据类
from random import randint, random  # 从 random 模块中导入 randint 和 random 函数，用于生成随机数

FOREST_LAND = 1000  # 定义 FOREST_LAND 常量，表示森林土地数量
INITIAL_LAND = FOREST_LAND + 1000  # 定义 INITIAL_LAND 常量，表示初始土地数量，包括森林土地和额外的1000单位土地
COST_OF_LIVING = 100  # 定义 COST_OF_LIVING 常量，表示生活成本
COST_OF_FUNERAL = 9  # 定义 COST_OF_FUNERAL 常量，表示葬礼成本
YEARS_IN_TERM = 8  # 定义 YEARS_IN_TERM 常量，表示任期年限
POLLUTION_CONTROL_FACTOR = 25  # 定义 POLLUTION_CONTROL_FACTOR 常量，表示污染控制因子
def ask_int(prompt) -> int:
    # 定义一个函数，用于提示用户输入一个整数，直到用户输入一个整数为止
    while True:
        try:
            return int(input(prompt))  # 尝试将用户输入的内容转换为整数，如果成功则返回该整数
        except ValueError:
            continue  # 如果用户输入的内容无法转换为整数，则继续提示用户输入直到输入一个整数为止


@dataclass
class GameState:
    rallods: int = -1  # 初始化游戏状态中的 rallods 属性为 -1
    countrymen: int = -1  # 初始化游戏状态中的 countrymen 属性为 -1
    land: int = INITIAL_LAND  # 初始化游戏状态中的 land 属性为 INITIAL_LAND 的值
    foreign_workers: int = 0  # 初始化游戏状态中的 foreign_workers 属性为 0
    years_in_office: int = 0  # 初始化游戏状态中的 years_in_office 属性为 0

    # previous year stats
    crop_loss_last_year: int = 0  # 初始化游戏状态中的 crop_loss_last_year 属性为 0

    # current year stats
    died_contrymen: int = 0  # 初始化 died_contrymen 变量为整数 0
    pollution_deaths: int = 0  # 初始化 pollution_deaths 变量为整数 0
    population_change: int = 0  # 初始化 population_change 变量为整数 0

    # current year - market situation (in rallods per square mile)
    planting_cost: int = -1  # 初始化 planting_cost 变量为整数 -1
    land_buy_price: int = -1  # 初始化 land_buy_price 变量为整数 -1

    tourism_earnings: int = 0  # 初始化 tourism_earnings 变量为整数 0

    def set_market_conditions(self) -> None:  # 定义一个方法 set_market_conditions，参数为 self，返回类型为 None
        self.land_buy_price = randint(95, 105)  # 设置 land_buy_price 为 95 到 105 之间的随机整数
        self.planting_cost = randint(10, 15)  # 设置 planting_cost 为 10 到 15 之间的随机整数

    @property
    def farmland(self) -> int:  # 定义一个属性 farmland，返回类型为整数
        return self.land - FOREST_LAND  # 返回 land 减去 FOREST_LAND 的值

    @property
    def settled_people(self) -> int:  # 定义一个属性 settled_people，返回类型为整数
        return self.countrymen - self.population_change
        # 返回当前国民数量减去人口变化的结果

    def sell_land(self, amount: int) -> None:
        assert amount <= self.farmland
        # 确保出售土地的数量不超过拥有的土地数量
        self.land -= amount
        # 减去出售的土地数量
        self.rallods += self.land_buy_price * amount
        # 增加相应的资金

    def distribute_rallods(self, distribute: int) -> None:
        self.rallods -= distribute
        # 减去分发的资金数量

    def spend_pollution_control(self, spend: int) -> None:
        self.rallods -= spend
        # 减去用于污染控制的资金数量

    def plant(self, sq_to_plant: int) -> None:
        self.rallods -= sq_to_plant * self.planting_cost
        # 减去用于种植的资金数量

    def print_status(self) -> None:
        print(f"\n\nYOU NOW HAVE {self.rallods} RALLODS IN THE TREASURY.")
        print(f"{int(self.countrymen)} COUNTRYMEN, ", end="")
        # 打印当前国民数量
        if self.foreign_workers > 0:
        # 如果有外来工人
# 打印外国工人数量和土地面积
print(f"{int(self.foreign_workers)} FOREIGN WORKERS, ", end="")
print(f"AND {self.land} SQ. MILES OF LAND.")
# 打印工业购买土地的价格
print(
    f"THIS YEAR INDUSTRY WILL BUY LAND FOR {self.land_buy_price} "
    "RALLODS PER SQUARE MILE."
)
# 打印种植成本
print(
    f"LAND CURRENTLY COSTS {self.planting_cost} RALLODS "
    "PER SQUARE MILE TO PLANT.\n"
)

# 处理死亡人数
def handle_deaths(
    self, distributed_rallods: int, pollution_control_spendings: int
) -> None:
    # 计算因饥饿而死亡的国民数量
    starved_countrymen = max(
        0, int(self.countrymen - distributed_rallods / COST_OF_LIVING)
    )

    # 如果有国民因饥饿而死亡，则打印死亡人数
    if starved_countrymen > 0:
        print(f"{starved_countrymen} COUNTRYMEN DIED OF STARVATION")
        # 计算因污染死亡的人数，根据当前土地数量和初始土地数量的差值乘以一个随机数
        self.pollution_deaths = int(random() * (INITIAL_LAND - self.land))
        # 如果污染控制支出大于等于污染控制因子，则重新计算死亡人数
        if pollution_control_spendings >= POLLUTION_CONTROL_FACTOR:
            self.pollution_deaths = int(
                self.pollution_deaths
                / (pollution_control_spendings / POLLUTION_CONTROL_FACTOR)
            )
        # 如果有人死于污染，则打印死亡人数和死因
        if self.pollution_deaths > 0:
            print(
                f"{self.pollution_deaths} COUNTRYMEN DIED OF CARBON-MONOXIDE "
                f"AND DUST INHALATION"
            )

        # 计算总共死亡的人数，包括因饥饿和污染死亡的人数
        self.died_contrymen = starved_countrymen + self.pollution_deaths
        # 如果有人死亡，则计算葬礼费用并打印相关信息
        if self.died_contrymen > 0:
            funeral_cost = self.died_contrymen * COST_OF_FUNERAL
            print(f"   YOU WERE FORCED TO SPEND {funeral_cost} RALLODS ON ")
            print("FUNERAL EXPENSES.")
            # 扣除葬礼费用后的剩余资金
            self.rallods -= funeral_cost
            # 如果资金不足，则进行相应处理
            if self.rallods < 0:
                print("   INSUFFICIENT RESERVES TO COVER COST - LAND WAS SOLD")
                # 打印消息，表示土地被卖出，因为储备不足以支付成本
                self.land += int(self.rallods / self.land_buy_price)
                # 增加土地数量，根据可用的货币数量和土地购买价格计算
                self.rallods = 0
                # 重置货币数量为0
            self.countrymen -= self.died_contrymen
            # 减少国民数量，根据死亡的国民数量

    def handle_tourist_trade(self) -> None:
        V1 = int(self.settled_people * 22 + random() * 500)
        # 计算V1，表示根据定居人口数量和随机数计算的值
        V2 = int((INITIAL_LAND - self.land) * 15)
        # 计算V2，表示根据初始土地数量和当前土地数量计算的值
        tourist_trade_earnings = 0
        # 初始化旅游贸易收入为0
        if V1 > V2:
            # 如果V1大于V2
            tourist_trade_earnings = V1 - V2
            # 计算旅游贸易收入
        print(f" YOU MADE {tourist_trade_earnings} RALLODS FROM TOURIST TRADE.")
        # 打印消息，表示从旅游贸易中获得的货币数量
        if V2 != 0 and not (V1 - V2 >= self.tourism_earnings):
            # 如果V2不等于0且V1减去V2不大于旅游收入
            print("   DECREASE BECAUSE ", end="")
            # 打印消息，表示减少的原因
            reason = randint(0, 10)
            # 生成一个0到10之间的随机数
            if reason <= 2:
                print("FISH POPULATION HAS DWINDLED DUE TO WATER POLLUTION.")
                # 如果随机数小于等于2，打印消息，表示鱼类数量因水污染而减少
            elif reason <= 4:
                print("AIR POLLUTION IS KILLING GAME BIRD POPULATION.")
                # 如果随机数小于等于4，打印消息，表示空气污染正在杀死野生鸟类数量
            elif reason <= 6:
                # 如果随机数小于等于6
                # 打印矿泉浴场因水污染而被破坏的消息
                print("MINERAL BATHS ARE BEING RUINED BY WATER POLLUTION.")
            # 如果原因小于等于8
            elif reason <= 8:
                # 打印令人不愉快的烟雾使日光浴者望而却步的消息
                print("UNPLEASANT SMOG IS DISCOURAGING SUN BATHERS.")
            else:
                # 打印由于烟雾灰尘而使酒店显得破旧的消息
                print("HOTELS ARE LOOKING SHABBY DUE TO SMOG GRIT.")

        # 注意：原始游戏中以下两行存在错误：
        # 计算旅游收入的绝对值并将其赋值给self.tourism_earnings
        self.tourism_earnings = abs(int(V1 - V2))
        # 将旅游收入加到rallods上
        self.rallods += self.tourism_earnings

    def handle_harvest(self, planted_sq: int) -> None:
        # 计算作物损失
        crop_loss = int((INITIAL_LAND - self.land) * ((random() + 1.5) / 2))
        # 如果有外国工人
        if self.foreign_workers != 0:
            # 打印种植的平方英里
            print(f"OF {planted_sq} SQ. MILES PLANTED,")
        # 如果种植的平方英里小于等于作物损失
        if planted_sq <= crop_loss:
            # 作物损失等于种植的平方英里
            crop_loss = planted_sq
        # 计算收获的作物
        harvested = int(planted_sq - crop_loss)
        # 打印收获的作物
        print(f" YOU HARVESTED {harvested} SQ. MILES OF CROPS.")
        # 不幸的收获更糟糕
        unlucky_harvesting_worse = crop_loss - self.crop_loss_last_year
        # 如果作物损失不为0
        if crop_loss != 0:
            print("   (DUE TO ", end="")  # 打印提示信息，指示接下来的情况
            if unlucky_harvesting_worse > 2:  # 如果不幸的收成更糟
                print("INCREASED ", end="")  # 打印提示信息，指示收成更糟
            print("AIR AND WATER POLLUTION FROM FOREIGN INDUSTRY.)")  # 打印提示信息，指示外国工业对空气和水的污染
        revenue = int((planted_sq - crop_loss) * (self.land_buy_price / 2))  # 计算收入
        print(f"MAKING {revenue} RALLODS.")  # 打印收入信息
        self.crop_loss_last_year = crop_loss  # 将去年的作物损失记录下来
        self.rallods += revenue  # 增加总收入

    def handle_foreign_workers(
        self,
        sm_sell_to_industry: int,
        distributed_rallods: int,
        polltion_control_spendings: int,
    ) -> None:
        foreign_workers_influx = 0  # 初始化外国工人流入数量
        if sm_sell_to_industry != 0:  # 如果卖给工业的数量不为0
            foreign_workers_influx = int(  # 计算外国工人流入数量
                sm_sell_to_industry + (random() * 10) - (random() * 20)
            )
            if self.foreign_workers <= 0:  # 如果外来工人数量小于等于0
                foreign_workers_influx = foreign_workers_influx + 20  # 增加外来工人数量
            print(f"{foreign_workers_influx} WORKERS CAME TO THE COUNTRY AND")  # 打印外来工人数量和提示信息

        surplus_distributed = distributed_rallods / COST_OF_LIVING - self.countrymen  # 计算分配剩余的资源
        population_change = int(
            (surplus_distributed / 10)  # 计算人口变化
            + (polltion_control_spendings / POLLUTION_CONTROL_FACTOR)  # 考虑污染控制支出对人口变化的影响
            - ((INITIAL_LAND - self.land) / 50)  # 考虑土地变化对人口变化的影响
            - (self.died_contrymen / 2)  # 考虑死亡对人口变化的影响
        )
        print(f"{abs(population_change)} COUNTRYMEN ", end="")  # 打印人口变化的绝对值和提示信息
        if population_change < 0:  # 如果人口变化为负数
            print("LEFT ", end="")  # 打印人口减少的提示信息
        else:
            print("CAME TO ", end="")  # 打印人口增加的提示信息
        print("THE ISLAND")  # 打印岛屿的提示信息
        self.countrymen += population_change  # 更新国民数量
        self.foreign_workers += int(foreign_workers_influx)  # 更新外来工人数量
    def handle_too_many_deaths(self) -> None:
        # 打印出国民死亡数量过多的警告信息
        print(f"\n\n\n{self.died_contrymen} COUNTRYMEN DIED IN ONE YEAR!!!!!")
        # 打印出因极度管理不善而被弹劾和罢免的信息
        print("\n\n\nDUE TO THIS EXTREME MISMANAGEMENT, YOU HAVE NOT ONLY")
        print("BEEN IMPEACHED AND THROWN OUT OF OFFICE, BUT YOU")
        # 生成一个随机数，表示额外的惩罚
        message = randint(0, 10)
        # 如果随机数小于等于3，打印出左眼被挖出的信息
        if message <= 3:
            print("ALSO HAD YOUR LEFT EYE GOUGED OUT!")
        # 如果随机数小于等于6，打印出声誉受损的信息
        if message <= 6:
            print("HAVE ALSO GAINED A VERY BAD REPUTATION.")
        # 如果随机数小于等于10，打印出被宣布为国家叛徒的信息
        if message <= 10:
            print("HAVE ALSO BEEN DECLARED NATIONAL FINK.")
        # 退出程序
        sys.exit()

    def handle_third_died(self) -> None:
        # 打印出人口死亡率过高的警告信息
        print()
        print()
        print("OVER ONE THIRD OF THE POPULTATION HAS DIED SINCE YOU")
        print("WERE ELECTED TO OFFICE. THE PEOPLE (REMAINING)")
        print("HATE YOUR GUTS.")
        # 结束游戏
        self.end_game()
        print("CONGRATULATIONS! YOU HAVE SUCCESSFULLY GOVERNED YOUR COUNTRY.")
        print("YOUR LEGACY WILL BE REMEMBERED FOR GENERATIONS TO COME.")
    else:
        print("YOU HAVE BEEN OVERTHROWN BY A MILITARY COUP. YOUR REIGN IS OVER.")
    sys.exit()
            print("YOU HAVE BEEN ASSASSINATED.")  # 打印玩家被暗杀的消息
        else:
            print("YOU HAVE BEEN THROWN OUT OF OFFICE AND ARE NOW")  # 打印玩家被赶出办公室的消息
            print("RESIDING IN PRISON.")  # 打印玩家被关进监狱的消息
        sys.exit()  # 退出程序

    def handle_congratulations(self) -> None:
        print("\n\nCONGRATULATIONS!!!!!!!!!!!!!!!!!!")  # 打印祝贺消息
        print(f"YOU HAVE SUCCESFULLY COMPLETED YOUR {YEARS_IN_TERM} YEAR TERM")  # 打印玩家成功完成任期的消息
        print("OF OFFICE. YOU WERE, OF COURSE, EXTREMELY LUCKY, BUT")  # 打印玩家成功的一些幸运
        print("NEVERTHELESS, IT'S QUITE AN ACHIEVEMENT. GOODBYE AND GOOD")  # 打印祝福玩家的消息
        print("LUCK - YOU'LL PROBABLY NEED IT IF YOU'RE THE TYPE THAT")  # 打印玩家可能需要好运的消息
        print("PLAYS THIS GAME.")  # 打印玩家玩这个游戏的消息
        sys.exit()  # 退出程序

def print_header() -> None:
    print(" " * 34 + "KING")  # 打印游戏标题
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")  # 打印游戏信息
def print_instructions() -> None:
    # 打印游戏说明
    print(
        f"""\n\n\nCONGRATULATIONS! YOU'VE JUST BEEN ELECTED PREMIER OF SETATS
DETINU, A SMALL COMMUNIST ISLAND 30 BY 70 MILES LONG. YOUR
JOB IS TO DECIDE UPON THE CONTRY'S BUDGET AND DISTRIBUTE
MONEY TO YOUR COUNTRYMEN FROM THE COMMUNAL TREASURY.
THE MONEY SYSTEM IS RALLODS, AND EACH PERSON NEEDS {COST_OF_LIVING}
RALLODS PER YEAR TO SURVIVE. YOUR COUNTRY'S INCOME COMES
FROM FARM PRODUCE AND TOURISTS VISITING YOUR MAGNIFICENT
FORESTS, HUNTING, FISHING, ETC. HALF YOUR LAND IS FARM LAND
WHICH ALSO HAS AN EXCELLENT MINERAL CONTENT AND MAY BE SOLD
TO FOREIGN INDUSTRY (STRIP MINING) WHO IMPORT AND SUPPORT
THEIR OWN WORKERS. CROPS COST BETWEEN 10 AND 15 RALLODS PER
SQUARE MILE TO PLANT.
YOUR GOAL IS TO COMPLETE YOUR {YEARS_IN_TERM} YEAR TERM OF OFFICE.
GOOD LUCK!"""
    )
def ask_how_many_sq_to_plant(state: GameState) -> int:
    # 询问用户要种植多少平方英里的土地
    while True:
        # 调用 ask_int 函数询问用户要种植多少平方英里的土地
        sq = ask_int("HOW MANY SQUARE MILES DO YOU WISH TO PLANT? ")
        # 如果用户输入的值小于0，则继续循环
        if sq < 0:
            continue
        # 如果用户输入的值大于每个国民可以种植的2倍平方英里数，则打印提示信息
        elif sq > 2 * state.countrymen:
            print("   SORRY, BUT EACH COUNTRYMAN CAN ONLY PLANT 2 SQ. MILES.")
        # 如果用户输入的值大于可用的农田面积，则打印提示信息
        elif sq > state.farmland:
            print(
                f"   SORRY, BUT YOU ONLY HAVE {state.farmland} "
                "SQ. MILES OF FARM LAND."
            )
        # 如果用户输入的值乘以种植成本大于国库中的货币数量，则打印提示信息
        elif sq * state.planting_cost > state.rallods:
            print(
                f"   THINK AGAIN. YOU'VE ONLY {state.rallods} RALLODS "
                "LEFT IN THE TREASURY."
            )
        # 如果用户输入的值符合条件，则返回该值
        else:
            return sq
# 定义一个函数，接受一个 GameState 对象作为参数，并返回一个整数
def ask_pollution_control(state: GameState) -> int:
    # 创建一个无限循环，直到满足条件才会退出循环
    while True:
        # 调用 ask_int 函数询问用户想要花费多少 rallods 来控制污染
        rallods = ask_int(
            "HOW MANY RALLODS DO YOU WISH TO SPEND ON POLLUTION CONTROL? "
        )
        # 如果用户输入的 rallods 大于状态对象中的 rallods 数量
        if rallods > state.rallods:
            # 打印提醒用户只有多少 rallods 可用
            print(f"   THINK AGAIN. YOU ONLY HAVE {state.rallods} RALLODS REMAINING.")
        # 如果用户输入的 rallods 小于 0
        elif rallods < 0:
            # 继续循环，等待用户输入正确的值
            continue
        # 如果用户输入的 rallods 符合条件
        else:
            # 返回用户输入的 rallods 数量
            return rallods


# 定义一个函数，接受一个 GameState 对象作为参数，并返回一个整数
def ask_sell_to_industry(state: GameState) -> int:
    # 初始化一个变量，用于记录是否出现了第一次错误
    had_first_err = False
    # 定义一个字符串，用于提示用户外国工业只会购买农田，因为森林土地不适合开采
    first = """(FOREIGN INDUSTRY WILL ONLY BUY FARM LAND BECAUSE
FOREST LAND IS UNECONOMICAL TO STRIP MINE DUE TO TREES,
THICKER TOP SOIL, ETC.)"""
    # 定义一个字符串，用于提示用户农田的数量
    err = f"""***  THINK AGAIN. YOU ONLY HAVE {state.farmland} SQUARE MILES OF FARM LAND."""
    while True:  # 创建一个无限循环，直到条件满足才会退出循环
        sm = input("HOW MANY SQUARE MILES DO YOU WISH TO SELL TO INDUSTRY? ")  # 从用户输入中获取要出售给工业的平方英里数
        try:  # 尝试将用户输入的内容转换为整数
            sm_sell = int(sm)  # 将用户输入的内容转换为整数
        except ValueError:  # 如果转换出错，捕获值错误异常
            if not had_first_err:  # 如果之前没有出现过错误
                print(first)  # 打印第一个错误信息
                had_first_err = True  # 将had_first_err标记为True，表示已经出现过第一个错误
            print(err)  # 打印错误信息
            continue  # 继续下一次循环
        if sm_sell > state.farmland:  # 如果要出售的平方英里数大于州的农田面积
            print(err)  # 打印错误信息
        elif sm_sell < 0:  # 如果要出售的平方英里数小于0
            continue  # 继续下一次循环
        else:  # 如果以上条件都不满足
            return sm_sell  # 返回要出售的平方英里数

def ask_distribute_rallods(state: GameState) -> int:  # 定义一个函数，接受一个GameState对象作为参数，并返回一个整数
    while True:  # 创建一个无限循环，直到条件满足才会退出循环
        rallods = ask_int(
            "HOW MANY RALLODS WILL YOU DISTRIBUTE AMONG YOUR COUNTRYMEN? "
        )  # 询问用户要分发给同胞的rallo数量，并将用户输入的整数赋值给rallods变量
        if rallods < 0:  # 如果rallods小于0
            continue  # 继续循环，重新询问用户输入
        elif rallods > state.rallods:  # 如果rallods大于state.rallods
            print(
                f"   THINK AGAIN. YOU'VE ONLY {state.rallods} RALLODS IN THE TREASURY"
            )  # 打印提醒信息，显示国库中的rallo数量
        else:  # 如果以上条件都不满足
            return rallods  # 返回用户输入的rallo数量


def resume() -> GameState:  # 定义一个返回GameState类型的函数resume
    while True:  # 无限循环
        years = ask_int("HOW MANY YEARS HAD YOU BEEN IN OFFICE WHEN INTERRUPTED? ")  # 询问用户在被中断时在任职多少年，并将用户输入的整数赋值给years变量
        if years < 0:  # 如果years小于0
            sys.exit()  # 退出程序
        if years >= YEARS_IN_TERM:  # 如果years大于等于YEARS_IN_TERM
            print(f"   COME ON, YOUR TERM IN OFFICE IS ONLY {YEARS_IN_TERM} YEARS.")  # 打印提醒信息，显示任期的年限
    else:
        break  # 结束循环
    treasury = ask_int("HOW MUCH DID YOU HAVE IN THE TREASURY? ")  # 询问用户国库中有多少钱
    if treasury < 0:  # 如果国库中的钱小于0
        sys.exit()  # 退出程序
    countrymen = ask_int("HOW MANY COUNTRYMEN? ")  # 询问国家有多少公民
    if countrymen < 0:  # 如果公民数量小于0
        sys.exit()  # 退出程序
    workers = ask_int("HOW MANY WORKERS? ")  # 询问有多少工人
    if workers < 0:  # 如果工人数量小于0
        sys.exit()  # 退出程序
    while True:  # 进入无限循环
        land = ask_int("HOW MANY SQUARE MILES OF LAND? ")  # 询问有多少平方英里的土地
        if land < 0:  # 如果土地数量小于0
            sys.exit()  # 退出程序
        if land > INITIAL_LAND:  # 如果土地数量大于初始土地数量
            farm_land = INITIAL_LAND - FOREST_LAND  # 计算农田土地数量
            print(f"   COME ON, YOU STARTED WITH {farm_land:,} SQ. MILES OF FARM LAND")  # 打印初始农田土地数量
            print(f"   AND {FOREST_LAND:,} SQ. MILES OF FOREST LAND.")  # 打印森林土地数量
        if land > FOREST_LAND:  # 如果土地数量大于森林土地数量
            break  # 结束循环，跳出当前循环体
    return GameState(  # 返回一个GameState对象
        rallods=treasury,  # 设置rallods属性为treasury变量的值
        countrymen=countrymen,  # 设置countrymen属性为countrymen变量的值
        foreign_workers=workers,  # 设置foreign_workers属性为workers变量的值
        years_in_office=years,  # 设置years_in_office属性为years变量的值
    )


def main() -> None:  # 定义一个名为main的函数，不返回任何值
    print_header()  # 调用print_header函数，打印标题
    want_instructions = input("DO YOU WANT INSTRUCTIONS? ").upper()  # 获取用户输入的指令并转换为大写
    if want_instructions == "AGAIN":  # 如果用户输入的指令是"AGAIN"
        state = resume()  # 调用resume函数，将返回值赋给state变量
    else:  # 否则
        state = GameState(  # 创建一个GameState对象
            rallods=randint(59000, 61000),  # 设置rallods属性为59000到61000之间的随机整数
            countrymen=randint(490, 510),  # 设置countrymen属性为490到510之间的随机整数
            planting_cost=randint(10, 15),  # 设置planting_cost属性为10到15之间的随机整数
        )
    if want_instructions != "NO":  # 如果用户不想要指导，则不打印指导信息
        print_instructions()  # 打印指导信息

    while True:  # 无限循环，直到条件被打破
        state.set_market_conditions()  # 设置市场条件
        state.print_status()  # 打印状态信息

        # 用户行动
        sm_sell_to_industry = ask_sell_to_industry(state)  # 询问用户是否卖给工业
        state.sell_land(sm_sell_to_industry)  # 卖出土地

        distributed_rallods = ask_distribute_rallods(state)  # 询问用户如何分配资源
        state.distribute_rallods(distributed_rallods)  # 分配资源

        planted_sq = ask_how_many_sq_to_plant(state)  # 询问用户要种植多少平方
        state.plant(planted_sq)  # 种植作物

        polltion_control_spendings = ask_pollution_control(state)  # 询问用户的污染控制支出
        state.spend_pollution_control(polltion_control_spendings)  # 支出污染控制费用

        # 运行一年
# 处理国内死亡人数和污染控制支出
state.handle_deaths(distributed_rallods, polltion_control_spendings)
# 处理外国工人
state.handle_foreign_workers(sm_sell_to_industry, distributed_rallods, polltion_control_spendings)
# 处理收获
state.handle_harvest(planted_sq)
# 处理旅游贸易
state.handle_tourist_trade()

# 如果死亡人数超过200，则处理过多死亡
if state.died_contrymen > 200:
    state.handle_too_many_deaths()
# 如果国民少于343，则处理第三次死亡
if state.countrymen < 343:
    state.handle_third_died()
# 如果每100个人中有5个以上的人死亡，并且死亡人数减去污染死亡人数大于等于2，则处理资金管理不善
elif (state.rallods / 100) > 5 and state.died_contrymen - state.pollution_deaths >= 2:
    state.handle_money_mismanagement()
# 如果外国工人数量大于国民数量，则处理外国人过多
if state.foreign_workers > state.countrymen:
    state.handle_too_many_foreigners()
# 如果在任期的倒数第二年，则处理祝贺
elif YEARS_IN_TERM - 1 == state.years_in_office:
    state.handle_congratulations()
# 否则
else:
            state.years_in_office += 1  # 增加state对象的years_in_office属性值
            state.died_contrymen = 0  # 将state对象的died_contrymen属性值设为0


if __name__ == "__main__":
    main()  # 如果当前脚本被直接执行，则调用main函数
```