# `basic-computer-games\53_King\python\king.py`

```
"""
KING

A strategy game where the player is the king.

Ported to Python by Martin Thoma in 2022
"""

# 导入 sys 模块
import sys
# 导入 dataclass 模块
from dataclasses import dataclass
# 导入 randint 和 random 函数
from random import randint, random

# 定义常量 FOREST_LAND
FOREST_LAND = 1000
# 定义常量 INITIAL_LAND
INITIAL_LAND = FOREST_LAND + 1000
# 定义常量 COST_OF_LIVING
COST_OF_LIVING = 100
# 定义常量 COST_OF_FUNERAL
COST_OF_FUNERAL = 9
# 定义常量 YEARS_IN_TERM
YEARS_IN_TERM = 8
# 定义常量 POLLUTION_CONTROL_FACTOR
POLLUTION_CONTROL_FACTOR = 25

# 定义函数 ask_int，提示用户输入整数
def ask_int(prompt) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            continue

# 定义数据类 GameState
@dataclass
class GameState:
    # 初始化属性
    rallods: int = -1
    countrymen: int = -1
    land: int = INITIAL_LAND
    foreign_workers: int = 0
    years_in_office: int = 0

    # previous year stats
    crop_loss_last_year: int = 0

    # current year stats
    died_contrymen: int = 0
    pollution_deaths: int = 0
    population_change: int = 0

    # current year - market situation (in rallods per square mile)
    planting_cost: int = -1
    land_buy_price: int = -1

    tourism_earnings: int = 0

    # 设置市场条件的方法
    def set_market_conditions(self) -> None:
        self.land_buy_price = randint(95, 105)
        self.planting_cost = randint(10, 15)

    # 计算 farmland 属性的方法
    @property
    def farmland(self) -> int:
        return self.land - FOREST_LAND

    # 计算 settled_people 属性的方法
    @property
    def settled_people(self) -> int:
        return self.countrymen - self.population_change

    # 出售土地的方法
    def sell_land(self, amount: int) -> None:
        assert amount <= self.farmland
        self.land -= amount
        self.rallods += self.land_buy_price * amount

    # 分发 rallods 的方法
    def distribute_rallods(self, distribute: int) -> None:
        self.rallods -= distribute

    # 花费污染控制的方法
    def spend_pollution_control(self, spend: int) -> None:
        self.rallods -= spend

    # 种植作物的方法
    def plant(self, sq_to_plant: int) -> None:
        self.rallods -= sq_to_plant * self.planting_cost
    # 打印当前状态信息
    def print_status(self) -> None:
        # 打印当前国库中的货币数量
        print(f"\n\nYOU NOW HAVE {self.rallods} RALLODS IN THE TREASURY.")
        # 打印当前国民数量
        print(f"{int(self.countrymen)} COUNTRYMEN, ", end="")
        # 如果有外国工人，则打印外国工人数量
        if self.foreign_workers > 0:
            print(f"{int(self.foreign_workers)} FOREIGN WORKERS, ", end="")
        # 打印当前土地面积
        print(f"AND {self.land} SQ. MILES OF LAND.")
        # 打印工业购买土地的价格
        print(
            f"THIS YEAR INDUSTRY WILL BUY LAND FOR {self.land_buy_price} "
            "RALLODS PER SQUARE MILE."
        )
        # 打印当前种植成本
        print(
            f"LAND CURRENTLY COSTS {self.planting_cost} RALLODS "
            "PER SQUARE MILE TO PLANT.\n"
        )

    # 处理死亡事件
    def handle_deaths(
        self, distributed_rallods: int, pollution_control_spendings: int
    # 定义一个方法，参数为self，distributed_rallods和pollution_control_spendings，返回空值
    def handle_deaths(self, distributed_rallods: int, pollution_control_spendings: int) -> None:
        # 计算因为分配的粮食不足导致的饿死的国民数量
        starved_countrymen = max(
            0, int(self.countrymen - distributed_rallods / COST_OF_LIVING)
        )

        # 如果有国民因为饥饿而死亡，则打印出死亡国民的数量
        if starved_countrymen > 0:
            print(f"{starved_countrymen} COUNTRYMEN DIED OF STARVATION")

        # 计算因为污染导致的死亡国民数量
        self.pollution_deaths = int(random() * (INITIAL_LAND - self.land))
        # 如果污染控制支出大于等于污染控制因子，则调整污染死亡数量
        if pollution_control_spendings >= POLLUTION_CONTROL_FACTOR:
            self.pollution_deaths = int(
                self.pollution_deaths
                / (pollution_control_spendings / POLLUTION_CONTROL_FACTOR)
            )
        # 如果有国民因为污染而死亡，则打印出死亡国民的数量
        if self.pollution_deaths > 0:
            print(
                f"{self.pollution_deaths} COUNTRYMEN DIED OF CARBON-MONOXIDE "
                f"AND DUST INHALATION"
            )

        # 计算总共死亡的国民数量
        self.died_contrymen = starved_countrymen + self.pollution_deaths
        # 如果有国民死亡，则计算葬礼费用，并打印出相关信息
        if self.died_contrymen > 0:
            funeral_cost = self.died_contrymen * COST_OF_FUNERAL
            print(f"   YOU WERE FORCED TO SPEND {funeral_cost} RALLODS ON ")
            print("FUNERAL EXPENSES.")
            self.rallods -= funeral_cost
            # 如果资金不足以支付葬礼费用，则卖出土地来弥补
            if self.rallods < 0:
                print("   INSUFFICIENT RESERVES TO COVER COST - LAND WAS SOLD")
                self.land += int(self.rallods / self.land_buy_price)
                self.rallods = 0
            # 减去死亡国民的数量
            self.countrymen -= self.died_contrymen
    # 处理旅游贸易的方法，没有返回值
    def handle_tourist_trade(self) -> None:
        # 计算旅游贸易收入，基于已定居人口和随机数
        V1 = int(self.settled_people * 22 + random() * 500)
        # 计算旅游贸易支出，基于初始土地和当前土地
        V2 = int((INITIAL_LAND - self.land) * 15)
        # 初始化旅游贸易收入
        tourist_trade_earnings = 0
        # 如果收入大于支出，计算旅游贸易收入
        if V1 > V2:
            tourist_trade_earnings = V1 - V2
        # 打印旅游贸易收入
        print(f" YOU MADE {tourist_trade_earnings} RALLODS FROM TOURIST TRADE.")
        # 如果支出不为零且收入减支出不大于旅游收入
        if V2 != 0 and not (V1 - V2 >= self.tourism_earnings):
            # 打印减少的原因
            print("   DECREASE BECAUSE ", end="")
            # 随机选择减少的原因
            reason = randint(0, 10)
            if reason <= 2:
                print("FISH POPULATION HAS DWINDLED DUE TO WATER POLLUTION.")
            elif reason <= 4:
                print("AIR POLLUTION IS KILLING GAME BIRD POPULATION.")
            elif reason <= 6:
                print("MINERAL BATHS ARE BEING RUINED BY WATER POLLUTION.")
            elif reason <= 8:
                print("UNPLEASANT SMOG IS DISCOURAGING SUN BATHERS.")
            else:
                print("HOTELS ARE LOOKING SHABBY DUE TO SMOG GRIT.")

        # 注意：原始游戏中以下两行存在错误
        # 计算旅游收入的绝对值，并赋值给旅游收入
        self.tourism_earnings = abs(int(V1 - V2))
        # 将旅游收入加到总收入中
        self.rallods += self.tourism_earnings
    # 处理收获，计算作物损失
    def handle_harvest(self, planted_sq: int) -> None:
        # 计算作物损失
        crop_loss = int((INITIAL_LAND - self.land) * ((random() + 1.5) / 2))
        # 如果有外国工人，则打印已种植的面积
        if self.foreign_workers != 0:
            print(f"OF {planted_sq} SQ. MILES PLANTED,")
        # 如果已种植的面积小于等于作物损失，则作物损失等于已种植的面积
        if planted_sq <= crop_loss:
            crop_loss = planted_sq
        # 计算收获的面积
        harvested = int(planted_sq - crop_loss)
        # 打印收获的面积
        print(f" YOU HARVESTED {harvested} SQ. MILES OF CROPS.")
        # 计算不幸的收获损失
        unlucky_harvesting_worse = crop_loss - self.crop_loss_last_year
        # 如果作物损失不为零，则打印由于外国工业的空气和水污染导致的情况
        if crop_loss != 0:
            print("   (DUE TO ", end="")
            if unlucky_harvesting_worse > 2:
                print("INCREASED ", end="")
            print("AIR AND WATER POLLUTION FROM FOREIGN INDUSTRY.)")
        # 计算收入
        revenue = int((planted_sq - crop_loss) * (self.land_buy_price / 2))
        # 打印收入
        print(f"MAKING {revenue} RALLODS.")
        # 更新去年的作物损失
        self.crop_loss_last_year = crop_loss
        # 更新总收入
        self.rallods += revenue

    # 处理外国工人
    def handle_foreign_workers(
        self,
        sm_sell_to_industry: int,
        distributed_rallods: int,
        polltion_control_spendings: int,
    # 定义一个方法，处理外来工人的流入情况
    def handle_foreign_workers(self, sm_sell_to_industry: int, distributed_rallods: int, polltion_control_spendings: int) -> None:
        # 初始化外来工人流入数量
        foreign_workers_influx = 0
        # 如果卖给工业的数量不为0
        if sm_sell_to_industry != 0:
            # 计算外来工人流入数量，基于卖给工业的数量和随机数
            foreign_workers_influx = int(
                sm_sell_to_industry + (random() * 10) - (random() * 20)
            )
            # 如果当前外来工人数量小于等于0
            if self.foreign_workers <= 0:
                # 增加外来工人流入数量
                foreign_workers_influx = foreign_workers_influx + 20
            # 打印外来工人流入数量
            print(f"{foreign_workers_influx} WORKERS CAME TO THE COUNTRY AND")

        # 计算分配的剩余资源与生活成本的比值
        surplus_distributed = distributed_rallods / COST_OF_LIVING - self.countrymen
        # 计算人口变化数量
        population_change = int(
            (surplus_distributed / 10)
            + (polltion_control_spendings / POLLUTION_CONTROL_FACTOR)
            - ((INITIAL_LAND - self.land) / 50)
            - (self.died_contrymen / 2)
        )
        # 打印人口变化数量
        print(f"{abs(population_change)} COUNTRYMEN ", end="")
        # 如果人口变化数量小于0
        if population_change < 0:
            # 打印人口减少信息
            print("LEFT ", end="")
        else:
            # 打印人口增加信息
            print("CAME TO ", end="")
        # 打印岛上人口变化信息
        print("THE ISLAND")
        # 更新国民数量
        self.countrymen += population_change
        # 增加外来工人数量
        self.foreign_workers += int(foreign_workers_influx)

    # 处理过多死亡情况的方法
    def handle_too_many_deaths(self) -> None:
        # 打印过多死亡的信息
        print(f"\n\n\n{self.died_contrymen} COUNTRYMEN DIED IN ONE YEAR!!!!!")
        print("\n\n\nDUE TO THIS EXTREME MISMANAGEMENT, YOU HAVE NOT ONLY")
        print("BEEN IMPEACHED AND THROWN OUT OF OFFICE, BUT YOU")
        # 随机生成一个数字
        message = randint(0, 10)
        # 根据随机数字打印不同的惩罚信息
        if message <= 3:
            print("ALSO HAD YOUR LEFT EYE GOUGED OUT!")
        if message <= 6:
            print("HAVE ALSO GAINED A VERY BAD REPUTATION.")
        if message <= 10:
            print("HAVE ALSO BEEN DECLARED NATIONAL FINK.")
        # 退出程序
        sys.exit()

    # 处理过多死亡情况的方法
    def handle_third_died(self) -> None:
        # 打印过多死亡的信息
        print()
        print()
        print("OVER ONE THIRD OF THE POPULTATION HAS DIED SINCE YOU")
        print("WERE ELECTED TO OFFICE. THE PEOPLE (REMAINING)")
        print("HATE YOUR GUTS.")
        # 结束游戏
        self.end_game()
    # 处理资金管理不善的情况，打印相关信息并终止程序
    def handle_money_mismanagement(self) -> None:
        # 打印相关信息
        print()
        print("MONEY WAS LEFT OVER IN THE TREASURY WHICH YOU DID")
        print("NOT SPEND. AS A RESULT, SOME OF YOUR COUNTRYMEN DIED")
        print("OF STARVATION. THE PUBLIC IS ENRAGED AND YOU HAVE")
        print("BEEN FORCED TO EITHER RESIGN OR COMMIT SUICIDE.")
        print("THE CHOICE IS YOURS.")
        print("IF YOU CHOOSE THE LATTER, PLEASE TURN OFF YOUR COMPUTER")
        # 终止程序
        sys.exit()
    
    # 处理外国人过多的情况，打印相关信息并调用结束游戏的方法
    def handle_too_many_foreigners(self) -> None:
        # 打印相关信息
        print("\n\nTHE NUMBER OF FOREIGN WORKERS HAS EXCEEDED THE NUMBER")
        print("OF COUNTRYMEN. AS A MINORITY, THEY HAVE REVOLTED AND")
        print("TAKEN OVER THE COUNTRY.")
        # 调用结束游戏的方法
        self.end_game()
    
    # 结束游戏的方法
    def end_game(self) -> None:
        # 如果随机数小于等于0.5，打印被暗杀的信息，否则打印被罢免的信息
        if random() <= 0.5:
            print("YOU HAVE BEEN ASSASSINATED.")
        else:
            print("YOU HAVE BEEN THROWN OUT OF OFFICE AND ARE NOW")
            print("RESIDING IN PRISON.")
        # 终止程序
        sys.exit()
    
    # 处理恭喜的情况，打印相关信息并终止程序
    def handle_congratulations(self) -> None:
        # 打印相关信息
        print("\n\nCONGRATULATIONS!!!!!!!!!!!!!!!!!!")
        print(f"YOU HAVE SUCCESFULLY COMPLETED YOUR {YEARS_IN_TERM} YEAR TERM")
        print("OF OFFICE. YOU WERE, OF COURSE, EXTREMELY LUCKY, BUT")
        print("NEVERTHELESS, IT'S QUITE AN ACHIEVEMENT. GOODBYE AND GOOD")
        print("LUCK - YOU'LL PROBABLY NEED IT IF YOU'RE THE TYPE THAT")
        print("PLAYS THIS GAME.")
        # 终止程序
        sys.exit()
# 打印游戏标题
def print_header() -> None:
    print(" " * 34 + "KING")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")


# 打印游戏说明
def print_instructions() -> None:
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


# 询问要种植多少平方英里的作物
def ask_how_many_sq_to_plant(state: GameState) -> int:
    while True:
        sq = ask_int("HOW MANY SQUARE MILES DO YOU WISH TO PLANT? ")
        if sq < 0:
            continue
        elif sq > 2 * state.countrymen:
            print("   SORRY, BUT EACH COUNTRYMAN CAN ONLY PLANT 2 SQ. MILES.")
        elif sq > state.farmland:
            print(
                f"   SORRY, BUT YOU ONLY HAVE {state.farmland} "
                "SQ. MILES OF FARM LAND."
            )
        elif sq * state.planting_cost > state.rallods:
            print(
                f"   THINK AGAIN. YOU'VE ONLY {state.rallods} RALLODS "
                "LEFT IN THE TREASURY."
            )
        else:
            return sq


# 询问是否进行污染控制
def ask_pollution_control(state: GameState) -> int:
    # 创建一个无限循环，直到条件满足才会退出
    while True:
        # 询问用户希望花费多少 rallods 来控制污染，接收用户输入的整数值
        rallods = ask_int(
            "HOW MANY RALLODS DO YOU WISH TO SPEND ON POLLUTION CONTROL? "
        )
        # 如果用户输入的 rallods 数量大于当前状态下的 rallods 数量
        if rallods > state.rallods:
            # 打印提醒信息，显示当前剩余的 rallods 数量
            print(f"   THINK AGAIN. YOU ONLY HAVE {state.rallods} RALLODS REMAINING.")
        # 如果用户输入的 rallods 数量小于 0
        elif rallods < 0:
            # 继续下一次循环
            continue
        # 如果用户输入的 rallods 数量符合条件
        else:
            # 返回用户输入的 rallods 数量
            return rallods
# 询问玩家要向工业出售多少平方英里的土地
def ask_sell_to_industry(state: GameState) -> int:
    # 标记是否已经出现第一个错误
    had_first_err = False
    # 第一个错误提示信息
    first = """(FOREIGN INDUSTRY WILL ONLY BUY FARM LAND BECAUSE
FOREST LAND IS UNECONOMICAL TO STRIP MINE DUE TO TREES,
THICKER TOP SOIL, ETC.)"""
    # 错误提示信息
    err = f"""***  THINK AGAIN. YOU ONLY HAVE {state.farmland} SQUARE MILES OF FARM LAND."""
    while True:
        # 询问玩家要出售多少平方英里的土地
        sm = input("HOW MANY SQUARE MILES DO YOU WISH TO SELL TO INDUSTRY? ")
        try:
            # 将输入的土地数量转换为整数
            sm_sell = int(sm)
        except ValueError:
            # 如果输入无法转换为整数，且之前没有出现过错误，则打印第一个错误提示信息
            if not had_first_err:
                print(first)
                had_first_err = True
            # 打印错误提示信息
            print(err)
            continue
        # 如果要出售的土地数量大于拥有的农田土地数量，则打印错误提示信息
        if sm_sell > state.farmland:
            print(err)
        # 如果要出售的土地数量小于0，则继续循环
        elif sm_sell < 0:
            continue
        # 否则返回要出售的土地数量
        else:
            return sm_sell


# 询问玩家要分配多少个 rallods 给同胞
def ask_distribute_rallods(state: GameState) -> int:
    while True:
        # 询问玩家要分配多少个 rallods 给同胞
        rallods = ask_int(
            "HOW MANY RALLODS WILL YOU DISTRIBUTE AMONG YOUR COUNTRYMEN? "
        )
        # 如果分配的 rallods 数量小于0，则继续循环
        if rallods < 0:
            continue
        # 如果分配的 rallods 数量大于国库中的 rallods 数量，则打印错误提示信息
        elif rallods > state.rallods:
            print(
                f"   THINK AGAIN. YOU'VE ONLY {state.rallods} RALLODS IN THE TREASURY"
            )
        # 否则返回分配的 rallods 数量
        else:
            return rallods


# 恢复游戏状态
def resume() -> GameState:
    while True:
        # 询问玩家在被中断时已经执政多少年
        years = ask_int("HOW MANY YEARS HAD YOU BEEN IN OFFICE WHEN INTERRUPTED? ")
        # 如果输入的年份小于0，则退出游戏
        if years < 0:
            sys.exit()
        # 如果输入的年份大于等于任期年限，则打印错误提示信息
        if years >= YEARS_IN_TERM:
            print(f"   COME ON, YOUR TERM IN OFFICE IS ONLY {YEARS_IN_TERM} YEARS.")
        else:
            break
    # 询问玩家国库中的资金数量
    treasury = ask_int("HOW MUCH DID YOU HAVE IN THE TREASURY? ")
    # 如果国库中的资金数量小于0，则退出游戏
    if treasury < 0:
        sys.exit()
    # 询问玩家同胞的数量
    countrymen = ask_int("HOW MANY COUNTRYMEN? ")
    # 如果同胞的数量小于0，则退出游戏
    if countrymen < 0:
        sys.exit()
    # 询问玩家工人的数量
    workers = ask_int("HOW MANY WORKERS? ")
    # 如果工人的数量小于0，则退出游戏
    if workers < 0:
        sys.exit()
    # 进入循环，直到条件不满足时退出
    while True:
        # 询问用户输入一个整数，表示土地的面积
        land = ask_int("HOW MANY SQUARE MILES OF LAND? ")
        # 如果输入的土地面积小于0，退出程序
        if land < 0:
            sys.exit()
        # 如果输入的土地面积大于初始土地面积
        if land > INITIAL_LAND:
            # 计算农田面积
            farm_land = INITIAL_LAND - FOREST_LAND
            # 打印提示信息，显示初始农田面积和森林面积
            print(f"   COME ON, YOU STARTED WITH {farm_land:,} SQ. MILES OF FARM LAND")
            print(f"   AND {FOREST_LAND:,} SQ. MILES OF FOREST LAND.")
        # 如果输入的土地面积大于森林面积，跳出循环
        if land > FOREST_LAND:
            break
    # 返回游戏状态对象，包括国库、国民、外国工人和执政年限
    return GameState(
        rallods=treasury,
        countrymen=countrymen,
        foreign_workers=workers,
        years_in_office=years,
    )
# 定义主函数，没有返回值
def main() -> None:
    # 打印标题
    print_header()
    # 询问用户是否需要说明
    want_instructions = input("DO YOU WANT INSTRUCTIONS? ").upper()
    # 如果用户需要再次说明，则恢复游戏状态
    if want_instructions == "AGAIN":
        state = resume()
    else:
        # 否则创建游戏状态对象
        state = GameState(
            rallods=randint(59000, 61000),
            countrymen=randint(490, 510),
            planting_cost=randint(10, 15),
        )
    # 如果用户不需要说明，则打印游戏说明
    if want_instructions != "NO":
        print_instructions()

    # 游戏循环
    while True:
        # 设置市场条件
        state.set_market_conditions()
        # 打印游戏状态
        state.print_status()

        # 用户行动
        sm_sell_to_industry = ask_sell_to_industry(state)
        state.sell_land(sm_sell_to_industry)

        distributed_rallods = ask_distribute_rallods(state)
        state.distribute_rallods(distributed_rallods)

        planted_sq = ask_how_many_sq_to_plant(state)
        state.plant(planted_sq)
        polltion_control_spendings = ask_pollution_control(state)
        state.spend_pollution_control(polltion_control_spendings)

        # 运行一年
        state.handle_deaths(distributed_rallods, polltion_control_spendings)
        state.handle_foreign_workers(
            sm_sell_to_industry, distributed_rallods, polltion_control_spendings
        )
        state.handle_harvest(planted_sq)
        state.handle_tourist_trade()

        # 处理游戏结束条件
        if state.died_contrymen > 200:
            state.handle_too_many_deaths()
        if state.countrymen < 343:
            state.handle_third_died()
        elif (
            state.rallods / 100
        ) > 5 and state.died_contrymen - state.pollution_deaths >= 2:
            state.handle_money_mismanagement()
        if state.foreign_workers > state.countrymen:
            state.handle_too_many_foreigners()
        elif YEARS_IN_TERM - 1 == state.years_in_office:
            state.handle_congratulations()
        else:
            state.years_in_office += 1
            state.died_contrymen = 0


if __name__ == "__main__":
    main()
```