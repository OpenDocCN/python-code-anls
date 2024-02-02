# `basic-computer-games\43_Hammurabi\python\hamurabi.py`

```py
# 从 random 模块中导入 random 和 seed 函数
from random import random, seed

# 生成一个 1 到 5 之间的随机整数
def gen_random() -> int:
    return int(random() * 5) + 1

# 打印错误信息
def bad_input_850() -> None:
    print("\nHAMURABI:  I CANNOT DO WHAT YOU WISH.")
    print("GET YOURSELF ANOTHER STEWARD!!!!!")

# 打印错误信息，显示当前的谷物数量
def bad_input_710(grain_bushels: int) -> None:
    print("HAMURABI:  THINK AGAIN.  YOU HAVE ONLY")
    print(f"{grain_bushels} BUSHELS OF GRAIN.  NOW THEN,")

# 打印错误信息，显示当前的土地数量
def bad_input_720(acres: float) -> None:
    print(f"HAMURABI:  THINK AGAIN.  YOU OWN ONLY {acres} ACRES.  NOW THEN,")

# 打印国家管理不善的错误信息
def national_fink() -> None:
    print("DUE TO THIS EXTREME MISMANAGEMENT YOU HAVE NOT ONLY")
    print("BEEN IMPEACHED AND THROWN OUT OF OFFICE BUT YOU HAVE")
    print("ALSO BEEN DECLARED NATIONAL FINK!!!!")

# 模拟 BASIC 输入，拒绝非数字值
def b_input(promptstring: str) -> int:
    """emulate BASIC input. It rejects non-numeric values"""
    x = input(promptstring)
    while x.isalpha():
        x = input("?REDO FROM START\n? ")
    return int(x)

# 主函数
def main() -> None:
    # 初始化随机数种子
    seed()
    title = "HAMURABI"
    # 将标题右对齐，填充空格
    title = title.rjust(32, " ")
    print(title)
    attribution = "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"
    # 将归属信息右对齐，填充空格
    attribution = attribution.rjust(15, " ")
    print(attribution)
    print("\n\n\n")
    print("TRY YOUR HAND AT GOVERNING ANCIENT SUMERIA")
    print("FOR A TEN-YEAR TERM OF OFFICE.\n")

    D1 = 0
    P1: float = 0
    year = 0
    population = 95
    grain_stores = 2800
    H = 3000
    eaten_rats = H - grain_stores
    bushels_per_acre = (
        3  # yield (amount of production from land). Reused as price per acre
    )
    acres = H / bushels_per_acre  # acres of land
    immigrants = 5
    plague = 1  # boolean for plague, also input for buy/sell land
    people = 0
    # 如果年份不等于99
    if year != 99:
        # 打印统计信息
        print("IN YOUR 10-YEAR TERM OF OFFICE,", P1, "PERCENT OF THE")
        print("POPULATION STARVED PER YEAR ON THE AVERAGE, I.E. A TOTAL OF")
        print(D1, "PEOPLE DIED!!")
        # 计算每个人的土地面积
        L = acres / population
        print("YOU STARTED WITH 10 ACRES PER PERSON AND ENDED WITH")
        print(L, "ACRES PER PERSON.\n")
        # 根据条件判断国家的状况
        if P1 > 33 or L < 7:
            national_fink()
        elif P1 > 10 or L < 9:
            print("YOUR HEAVY-HANDED PERFORMANCE SMACKS OF NERO AND IVAN IV.")
            print("THE PEOPLE (REMAINING) FIND YOU AN UNPLEASANT RULER, AND,")
            print("FRANKLY, HATE YOUR GUTS!!")
        elif P1 > 3 or L < 10:
            print("YOUR PERFORMANCE COULD HAVE BEEN SOMEWHAT BETTER, BUT")
            print(
                "REALLY WASN'T TOO BAD AT ALL. ",
                int(population * 0.8 * random()),
                "PEOPLE",
            )
            print("WOULD DEARLY LIKE TO SEE YOU ASSASSINATED BUT WE ALL HAVE OUR")
            print("TRIVIAL PROBLEMS.")
        else:
            print("A FANTASTIC PERFORMANCE!!!  CHARLEMANGE, DISRAELI, AND")
            print("JEFFERSON COMBINED COULD NOT HAVE DONE BETTER!\n")
        # 发出警报声音
        for _ in range(1, 10):
            print("\a")

    # 打印结束语
    print("\nSO LONG FOR NOW.\n")
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```