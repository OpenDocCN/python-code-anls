# `basic-computer-games\53_King\python\king.py`

```

"""
KING

A strategy game where the player is the king.

Ported to Python by Martin Thoma in 2022
"""

# 导入所需的模块
import sys
from dataclasses import dataclass
from random import randint, random

# 定义常量
FOREST_LAND = 1000
INITIAL_LAND = FOREST_LAND + 1000
COST_OF_LIVING = 100
COST_OF_FUNERAL = 9
YEARS_IN_TERM = 8
POLLUTION_CONTROL_FACTOR = 25

# 定义函数，提示用户输入整数
def ask_int(prompt) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            continue

# 定义数据类，打印游戏标题
@dataclass
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

# 提示用户输入要种植的平方英里数
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

# 提示用户输入用于污染控制的资金
def ask_pollution_control(state: GameState) -> int:
    while True:
        rallods = ask_int(
            "HOW MANY RALLODS DO YOU WISH TO SPEND ON POLLUTION CONTROL? "
        )
        if rallods > state.rallods:
            print(f"   THINK AGAIN. YOU ONLY HAVE {state.rallods} RALLODS REMAINING.")
        elif rallods < 0:
            continue
        else:
            return rallods

# 提示用户输入要卖给工业的土地面积
def ask_sell_to_industry(state: GameState) -> int:
    had_first_err = False
    first = """(FOREIGN INDUSTRY WILL ONLY BUY FARM LAND BECAUSE
FOREST LAND IS UNECONOMICAL TO STRIP MINE DUE TO TREES,
THICKER TOP SOIL, ETC.)"""
    err = f"""***  THINK AGAIN. YOU ONLY HAVE {state.farmland} SQUARE MILES OF FARM LAND."""
    while True:
        sm = input("HOW MANY SQUARE MILES DO YOU WISH TO SELL TO INDUSTRY? ")
        try:
            sm_sell = int(sm)
        except ValueError:
            if not had_first_err:
                print(first)
                had_first_err = True
            print(err)
            continue
        if sm_sell > state.farmland:
            print(err)
        elif sm_sell < 0:
            continue
        else:
            return sm_sell

# 提示用户输入要分配的资金
def ask_distribute_rallods(state: GameState) -> int:
    while True:
        rallods = ask_int(
            "HOW MANY RALLODS WILL YOU DISTRIBUTE AMONG YOUR COUNTRYMEN? "
        )
        if rallods < 0:
            continue
        elif rallods > state.rallods:
            print(
                f"   THINK AGAIN. YOU'VE ONLY {state.rallods} RALLODS IN THE TREASURY"
            )
        else:
            return rallods

# 恢复游戏
def resume() -> GameState:
    while True:
        years = ask_int("HOW MANY YEARS HAD YOU BEEN IN OFFICE WHEN INTERRUPTED? ")
        if years < 0:
            sys.exit()
        if years >= YEARS_IN_TERM:
            print(f"   COME ON, YOUR TERM IN OFFICE IS ONLY {YEARS_IN_TERM} YEARS.")
        else:
            break
    treasury = ask_int("HOW MUCH DID YOU HAVE IN THE TREASURY? ")
    if treasury < 0:
        sys.exit()
    countrymen = ask_int("HOW MANY COUNTRYMEN? ")
    if countrymen < 0:
        sys.exit()
    workers = ask_int("HOW MANY WORKERS? ")
    if workers < 0:
        sys.exit()
    while True:
        land = ask_int("HOW MANY SQUARE MILES OF LAND? ")
        if land < 0:
            sys.exit()
        if land > INITIAL_LAND:
            farm_land = INITIAL_LAND - FOREST_LAND
            print(f"   COME ON, YOU STARTED WITH {farm_land:,} SQ. MILES OF FARM LAND")
            print(f"   AND {FOREST_LAND:,} SQ. MILES OF FOREST LAND.")
        if land > FOREST_LAND:
            break
    return GameState(
        rallods=treasury,
        countrymen=countrymen,
        foreign_workers=workers,
        years_in_office=years,
    )

# 主函数
def main() -> None:
    print_header()
    want_instructions = input("DO YOU WANT INSTRUCTIONS? ").upper()
    if want_instructions == "AGAIN":
        state = resume()
    else:
        state = GameState(
            rallods=randint(59000, 61000),
            countrymen=randint(490, 510),
            planting_cost=randint(10, 15),
        )
    if want_instructions != "NO":
        print_instructions()

    while True:
        state.set_market_conditions()
        state.print_status()

        # 用户的行动
        sm_sell_to_industry = ask_sell_to_industry(state)
        state.sell_land(sm_sell_to_industry)

        distributed_rallods = ask_distribute_rallods(state)
        state.distribute_rallods(distributed_rallods)

        planted_sq = ask_how_many_sq_to_plant(state)
        state.plant(planted_sq)
        polltion_control_spendings = ask_pollution_control(state)
        state.spend_pollution_control(pollution_control_spendings)

        # 运行一年
        state.handle_deaths(distributed_rallods, polltion_control_spendings)
        state.handle_foreign_workers(
            sm_sell_to_industry, distributed_rallods, polltion_control_spendings
        )
        state.handle_harvest(planted_sq)
        state.handle_tourist_trade()

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