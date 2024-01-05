# `d:/src/tocomm/basic-computer-games\43_Hammurabi\python\hamurabi.py`

```
from random import random, seed  # 从 random 模块中导入 random 和 seed 函数


def gen_random() -> int:  # 定义一个返回整数的函数 gen_random
    return int(random() * 5) + 1  # 返回一个 1 到 5 之间的随机整数


def bad_input_850() -> None:  # 定义一个不返回任何值的函数 bad_input_850
    print("\nHAMURABI:  I CANNOT DO WHAT YOU WISH.")  # 打印提示信息
    print("GET YOURSELF ANOTHER STEWARD!!!!!")  # 打印提示信息


def bad_input_710(grain_bushels: int) -> None:  # 定义一个不返回任何值的函数 bad_input_710，接受一个整数参数 grain_bushels
    print("HAMURABI:  THINK AGAIN.  YOU HAVE ONLY")  # 打印提示信息
    print(f"{grain_bushels} BUSHELS OF GRAIN.  NOW THEN,")  # 打印提示信息


def bad_input_720(acres: float) -> None:  # 定义一个不返回任何值的函数 bad_input_720，接受一个浮点数参数 acres
    print(f"HAMURABI:  THINK AGAIN.  YOU OWN ONLY {acres} ACRES.  NOW THEN,")  # 打印提示信息
def national_fink() -> None:
    # 打印声明
    print("DUE TO THIS EXTREME MISMANAGEMENT YOU HAVE NOT ONLY")
    print("BEEN IMPEACHED AND THROWN OUT OF OFFICE BUT YOU HAVE")
    print("ALSO BEEN DECLARED NATIONAL FINK!!!!")


def b_input(promptstring: str) -> int:
    """emulate BASIC input. It rejects non-numeric values"""
    # 模拟BASIC输入，拒绝非数字值
    x = input(promptstring)
    while x.isalpha():
        x = input("?REDO FROM START\n? ")
    return int(x)


def main() -> None:
    # 生成随机种子
    seed()
    title = "HAMURABI"
    # 将标题右对齐，总长度为32，用空格填充
    title = title.rjust(32, " ")
    print(title)
    attribution = "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"  # 定义变量 attribution，存储字符串
    attribution = attribution.rjust(15, " ")  # 将 attribution 右对齐，总长度为 15，不足部分用空格填充
    print(attribution)  # 打印 attribution 变量的值
    print("\n\n\n")  # 打印三个换行符

    print("TRY YOUR HAND AT GOVERNING ANCIENT SUMERIA")  # 打印提示信息
    print("FOR A TEN-YEAR TERM OF OFFICE.\n")  # 打印提示信息并换行

    D1 = 0  # 定义变量 D1，初始化为 0
    P1: float = 0  # 定义变量 P1，指定类型为 float，初始化为 0
    year = 0  # 定义变量 year，初始化为 0
    population = 95  # 定义变量 population，初始化为 95
    grain_stores = 2800  # 定义变量 grain_stores，初始化为 2800
    H = 3000  # 定义变量 H，初始化为 3000
    eaten_rats = H - grain_stores  # 计算变量 eaten_rats 的值
    bushels_per_acre = 3  # 定义变量 bushels_per_acre，初始化为 3，表示每英亩土地的产量，后面会被重用为每英亩土地的价格
    acres = H / bushels_per_acre  # 计算变量 acres 的值，表示土地的英亩数
    immigrants = 5  # 定义变量 immigrants，初始化为 5
    plague = 1  # 定义变量 plague，初始化为 1，表示是否有瘟疫，也作为买卖土地的输入
    people = 0  # 初始化变量 people 为 0

    while year < 11:  # 行270. 主循环。当年份小于11时执行循环
        print("\n\n\nHAMURABI:  I BEG TO REPORT TO YOU")  # 打印提示信息
        year = year + 1  # 年份加1
        print(
            "IN YEAR",
            year,
            ",",
            people,
            "PEOPLE STARVED,",
            immigrants,
            "CAME TO THE CITY,",
        )  # 打印年份、饥饿人数、移民人数的信息
        population = population + immigrants  # 城市人口增加移民人数

        if plague == 0:  # 如果没有瘟疫
            population = int(population / 2)  # 人口减半
            print("A HORRIBLE PLAGUE STRUCK!  HALF THE PEOPLE DIED.")  # 打印瘟疫信息
        print("POPULATION IS NOW", population)  # 打印当前人口数量
        print("THE CITY NOW OWNS", acres, "ACRES.")  # 打印城市拥有的土地面积
        print("YOU HARVESTED", bushels_per_acre, "BUSHELS PER ACRE.")  # 打印每英亩收获的小麦数量
        print("THE RATS ATE", eaten_rats, "BUSHELS.")  # 打印老鼠吃掉的小麦数量
        print("YOU NOW HAVE ", grain_stores, "BUSHELS IN STORE.\n")  # 打印当前存储的小麦数量
        C = int(10 * random())  # 生成一个1到10之间的随机数
        bushels_per_acre = C + 17  # 计算每英亩收获的小麦数量
        print("LAND IS TRADING AT", bushels_per_acre, "BUSHELS PER ACRE.")  # 打印土地的交易价格

        plague = -99  # 用于跟踪状态的虚拟值
        while plague == -99:  # 总是至少运行一次循环
            plague = b_input("HOW MANY ACRES DO YOU WISH TO BUY? ")  # 获取用户输入的购买土地的数量
            if plague < 0:  # 如果输入小于0
                plague = -1  # 避免Q=-99的特殊情况
                bad_input_850()  # 调用错误处理函数
                year = 99  # 跳出主循环并退出
            elif bushels_per_acre * plague > grain_stores:  # 如果无法负担
                bad_input_710(grain_stores)  # 调用错误处理函数
                plague = -99  # 给用户第二次机会输入
            elif (
                bushels_per_acre * plague <= grain_stores
            ):  # normal case, can afford it
                acres = acres + plague  # increase the number of acres by Q
                grain_stores = (
                    grain_stores - bushels_per_acre * plague
                )  # decrease the amount of grain in store to pay for it
                C = 0  # WTF is C for?

        if plague == 0 and year != 99:  # maybe you want to sell some land?
            plague = -99
            while plague == -99:
                plague = b_input("HOW MANY ACRES DO YOU WISH TO SELL? ")
                if plague < 0:
                    bad_input_850()
                    year = 99  # jump out of main loop and exit
                elif plague <= acres:  # normal case
                    acres = acres - plague  # reduce the acres
                    grain_stores = (
                        grain_stores + bushels_per_acre * plague
                    )  # add to grain stores
```

在这段代码中，需要添加注释来解释每个语句的作用和意图。例如：

- `bushels_per_acre * plague <= grain_stores`: 检查是否有足够的粮食来购买土地，如果条件成立则执行下面的代码块。
- `acres = acres + plague`: 增加土地的数量。
- `grain_stores = (grain_stores - bushels_per_acre * plague)`: 减少存储的粮食数量以支付购买土地的费用。
- `C = 0`: 设置变量C的值为0，但注释中提到"WTF is C for?"，表示对变量C的用途存在疑惑。

另外，对于后面的if语句和while循环，也需要添加注释来解释其作用和条件。
C = 0  # still don't know what C is for
# 初始化变量C，但是不清楚它的作用是什么

else:  # Q>A error!
    bad_input_720(acres)
    plague = -99  # reloop
# 如果不满足上面的条件，调用bad_input_720函数，然后重新循环

plague = -99
while plague == -99 and year != 99:
    plague = b_input("HOW MANY BUSHELS DO YOU WISH TO FEED YOUR PEOPLE? ")
    if plague < 0:
        bad_input_850()
        year = 99  # jump out of main loop and exit
    # REM *** TRYING TO USE MORE GRAIN THAN IS IN SILOS?
    elif plague > grain_stores:
        bad_input_710(grain_stores)
        plague = -99  # try again!
    else:  # we're good. do the transaction
        grain_stores = grain_stores - plague  # remove the grain from the stores
        C = 1  # set the speed of light to 1. jk
# 如果输入的plague小于0，调用bad_input_850函数并退出主循环
# 如果输入的plague大于粮食库存，调用bad_input_710函数并重新尝试
# 否则，从库存中减去相应数量的粮食，并将C设置为1
        print("\n")  # 打印空行

        people = -99  # 初始化变量people为-99，用于强制至少执行一次循环

        while people == -99 and year != 99:  # 当people为-99且year不等于99时执行循环
            people = b_input("HOW MANY ACRES DO YOU WISH TO PLANT WITH SEED? ")  # 从用户输入获取种子需要种植的土地面积

            if people < 0:  # 如果输入的土地面积小于0
                bad_input_850()  # 调用bad_input_850函数
                year = 99  # 设置year为99，跳出主循环并退出
            elif people > 0:  # 如果输入的土地面积大于0
                if people > acres:  # 如果输入的土地面积大于拥有的土地面积
                    # REM *** TRYING TO PLANT MORE ACRES THAN YOU OWN?
                    bad_input_720(acres)  # 调用bad_input_720函数，传入拥有的土地面积作为参数
                    people = -99  # 重置people为-99
                elif int(people / 2) > grain_stores:  # 如果种植所需的粮食数量大于存储的粮食数量的一半
                    # REM *** ENOUGH GRAIN FOR SEED?
                    bad_input_710(grain_stores)  # 调用bad_input_710函数，传入存储的粮食数量作为参数
                    people = -99  # 重置people为-99
                elif people > 10 * population:  # 如果种植所需的人手数量大于当前人口的10倍
                    # REM *** ENOUGH PEOPLE TO TEND THE CROPS?
                    print("BUT YOU HAVE ONLY",  # 打印提示信息
        C = gen_random()  # 生成一个随机数并赋值给变量C
        bushels_per_acre = C  # 将变量C的值赋给每英亩收成的小麦数量
        H = people * bushels_per_acre  # 计算总收成，即人口数量乘以每英亩收成的小麦数量
        eaten_rats = 0  # 初始化被老鼠吃掉的小麦数量为0

        C = gen_random()  # 生成一个随机数并赋值给变量C
        if int(C / 2) == C / 2:  # 如果C除以2的整数部分等于C除以2，即C为偶数，有50%的概率
            eaten_rats = int(
                grain_stores / C
            )  # 计算被老鼠吃掉的小麦数量，基于之前生成的随机数
        grain_stores = grain_stores - eaten_rats + H  # 从粮食库存中扣除被老鼠吃掉的粮食，再加上新的收获

        C = gen_random()
        # REM *** LET'S HAVE SOME BABIES
        immigrants = int(C * (20 * acres + grain_stores) / population / 100 + 1)
        # REM *** HOW MANY PEOPLE HAD FULL TUMMIES?
        C = int(plague / 20)
        # REM *** HORROS, A 15% CHANCE OF PLAGUE
        # 是 HORRORS，但是保留了
        plague = int(10 * (2 * random() - 0.3))
        if (
            population >= C and year != 99
        ):  # 如果有一些人没有饱腹...
            # REM *** STARVE ENOUGH FOR IMPEACHMENT?
            people = population - C
            if people > 0.45 * population:
                print("\nYOU STARVED", people, "PEOPLE IN ONE YEAR!!!")
                national_fink()
                year = 99  # 退出循环
            P1 = ((year - 1) * P1 + people * 100 / population) / year
            population = C  # 将变量C的值赋给population，假设C是人口数量
            D1 = D1 + people  # 将people的值加到D1上，假设people是死亡人数

    if year != 99:  # 如果年份不等于99
        print("IN YOUR 10-YEAR TERM OF OFFICE,", P1, "PERCENT OF THE")  # 打印在你的十年任期内，百分之P1的人口平均每年挨饿
        print("POPULATION STARVED PER YEAR ON THE AVERAGE, I.E. A TOTAL OF")  # 打印平均每年有多少人挨饿，即总共有多少人死亡
        print(D1, "PEOPLE DIED!!")  # 打印D1个人死亡
        L = acres / population  # 计算每个人拥有的土地面积
        print("YOU STARTED WITH 10 ACRES PER PERSON AND ENDED WITH")  # 打印你开始时每个人拥有10英亩土地，结束时每个人拥有
        print(L, "ACRES PER PERSON.\n")  # 打印每个人拥有的土地面积
        if P1 > 33 or L < 7:  # 如果P1大于33或者L小于7
            national_fink()  # 调用national_fink函数
        elif P1 > 10 or L < 9:  # 如果P1大于10或者L小于9
            print("YOUR HEAVY-HANDED PERFORMANCE SMACKS OF NERO AND IVAN IV.")  # 打印你的高压表现让人想起尼禄和伊凡四世
            print("THE PEOPLE (REMAINING) FIND YOU AN UNPLEASANT RULER, AND,")  # 打印人民（剩下的）发现你是一个讨厌的统治者，并且
            print("FRANKLY, HATE YOUR GUTS!!")  # 打印坦率地，讨厌你
        elif P1 > 3 or L < 10:  # 如果P1大于3或者L小于10
            print("YOUR PERFORMANCE COULD HAVE BEEN SOMEWHAT BETTER, BUT")  # 打印你的表现可能会更好一些，但是
            print("REALLY WASN'T TOO BAD AT ALL. ",  # 打印实际上一点也不糟糕
# 定义一个名为 main 的函数，用于程序的主要逻辑
def main():
    # 打印欢迎信息
    print("WELCOME TO THE AUTOMATED POST OFFICE!")
    # 调用 send_mail 函数，传入收件人和寄件人信息
    send_mail(
        "HONORED RECIPIENT",
        "YOUR FRIEND",
    )
    # 打印结束语
    print("\nSO LONG FOR NOW.\n")

# 如果当前脚本作为主程序运行，则执行 main 函数
if __name__ == "__main__":
    main()
```