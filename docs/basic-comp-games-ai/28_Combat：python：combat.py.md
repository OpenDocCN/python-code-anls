# `d:/src/tocomm/basic-computer-games\28_Combat\python\combat.py`

```
MAX_UNITS = 72000  # 设置最大单位数为72000
plane_crash_win = False  # 初始化飞机坠毁胜利标志为False
usr_army = 0  # 初始化玩家陆军数量为0
usr_navy = 0  # 初始化玩家海军数量为0
usr_air = 0  # 初始化玩家空军数量为0
cpu_army = 30000  # 初始化CPU陆军数量为30000
cpu_navy = 20000  # 初始化CPU海军数量为20000
cpu_air = 22000  # 初始化CPU空军数量为22000

def show_intro() -> None:
    global MAX_UNITS  # 声明引用全局变量MAX_UNITS

    print(" " * 32 + "COMBAT")  # 打印游戏标题
    print(" " * 14 + "CREATIVE COMPUTING MORRISTOWN, NEW JERSEY")  # 打印游戏信息
    print("\n\n")  # 打印空行
    print("I AM AT WAR WITH YOU.")  # 打印游戏介绍
    print("WE HAVE " + str(MAX_UNITS) + " SOLDIERS APIECE.")  # 打印最大单位数
def get_forces() -> None:
    global usr_army, usr_navy, usr_air  # 声明全局变量 usr_army, usr_navy, usr_air

    while True:  # 进入无限循环
        print("DISTRIBUTE YOUR FORCES.")  # 打印提示信息
        print("              ME              YOU")  # 打印表头
        print("ARMY           " + str(cpu_army) + "        ? ", end="")  # 打印军队数量提示信息
        usr_army = int(input())  # 获取用户输入的军队数量
        print("NAVY           " + str(cpu_navy) + "        ? ", end="")  # 打印海军数量提示信息
        usr_navy = int(input())  # 获取用户输入的海军数量
        print("A. F.          " + str(cpu_air) + "        ? ", end="")  # 打印空军数量提示信息
        usr_air = int(input())  # 获取用户输入的空军数量
        if (usr_army + usr_navy + usr_air) <= MAX_UNITS:  # 判断用户输入的总兵力是否小于等于最大兵力
            break  # 如果是，则跳出循环


def attack_first() -> None:
    global usr_army, usr_navy, usr_air  # 声明全局变量 usr_army, usr_navy, usr_air
    global cpu_army, cpu_navy, cpu_air  # 声明全局变量 cpu_army, cpu_navy, cpu_air
    num_units = 0  # 初始化变量 num_units 为 0，用于存储输入的单位数量
    unit_type = 0  # 初始化变量 unit_type 为 0，用于存储输入的单位类型

    while True:  # 进入无限循环，直到用户输入合法的单位类型
        print("YOU ATTACK FIRST. TYPE (1) FOR ARMY; (2) FOR NAVY;")
        print("AND (3) FOR AIR FORCE.")
        print("?", end=" ")  # 输出提示信息，等待用户输入
        unit_type = int(input())  # 获取用户输入的单位类型
        if not (unit_type < 1 or unit_type > 3):  # 如果输入的单位类型合法，则跳出循环
            break

    while True:  # 进入无限循环，直到用户输入合法的单位数量
        print("HOW MANY MEN")
        print("?", end=" ")  # 输出提示信息，等待用户输入
        num_units = int(input())  # 获取用户输入的单位数量
        if not (
            (num_units < 0)  # 如果输入的单位数量小于 0，则继续循环
            or ((unit_type == 1) and (num_units > usr_army))  # 如果输入的是陆军单位且数量大于用户拥有的陆军数量，则继续循环
            or ((unit_type == 2) and (num_units > usr_navy))  # 如果输入的是海军单位且数量大于用户拥有的海军数量，则继续循环
            or ((unit_type == 3) and (num_units > usr_air))  # 如果输入的是空军单位且数量大于用户拥有的空军数量，则继续循环
        ):
            break  # 如果用户输入的单位类型不在1到3之间，跳出循环

    if unit_type == 1:  # 如果用户选择的单位类型是1
        if num_units < (usr_army / 3):  # 如果用户输入的单位数量小于用户军队数量的三分之一
            print("YOU LOST " + str(num_units) + " MEN FROM YOUR ARMY.")  # 打印用户失去的军队数量
            usr_army = usr_army - num_units  # 更新用户军队数量
        elif num_units < (2 * usr_army / 3):  # 如果用户输入的单位数量小于用户军队数量的三分之二
            print(
                "YOU LOST "
                + str(int(num_units / 3))
                + " MEN, BUT I LOST "
                + str(int(2 * cpu_army / 3))
            )  # 打印用户失去的军队数量和电脑失去的军队数量
            usr_army = int(usr_army - (num_units / 3))  # 更新用户军队数量
            cpu_army = 0  # 更新电脑军队数量为0
        else:  # 如果用户输入的单位数量大于等于用户军队数量的三分之二
            print("YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO")
            print("OF YOUR AIR FORCE BASES AND 3 ARMY BASES.")
            usr_army = int(usr_army / 3)  # 更新用户军队数量为原来的三分之一
            usr_air = int(usr_air / 3)  # 将用户空军数量的1/3赋值给usr_air
            cpu_navy = int(2 * cpu_navy / 3)  # 将CPU海军数量的2/3赋值给cpu_navy
    elif unit_type == 2:  # 如果unit_type等于2
        if num_units < cpu_navy / 3:  # 如果用户单位数量小于CPU海军数量的1/3
            print("YOUR ATTACK WAS STOPPED!")  # 打印"YOUR ATTACK WAS STOPPED!"
            usr_navy = usr_navy - num_units  # 用户海军数量减去单位数量
        elif num_units < 2 * cpu_navy / 3:  # 如果用户单位数量小于CPU海军数量的2/3
            print("YOU DESTROYED " + str(int(2 * cpu_navy / 3)) + " OF MY ARMY.")  # 打印"YOU DESTROYED " + str(int(2 * cpu_navy / 3)) + " OF MY ARMY."
            cpu_navy = int(cpu_navy / 3)  # 将CPU海军数量的1/3赋值给cpu_navy
        else:  # 否则
            print("YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO")  # 打印"YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO"
            print("OF YOUR AIR FORCE BASES AND 3 ARMY BASES.")  # 打印"OF YOUR AIR FORCE BASES AND 3 ARMY BASES."
            usr_army = int(usr_army / 3)  # 将用户陆军数量的1/3赋值给usr_army
            usr_air = int(usr_air / 3)  # 将用户空军数量的1/3赋值给usr_air
            cpu_navy = int(2 * cpu_navy / 3)  # 将CPU海军数量的2/3赋值给cpu_navy
    elif unit_type == 3:  # 如果unit_type等于3
        if num_units < usr_air / 3:  # 如果用户单位数量小于用户空军数量的1/3
            print("YOUR ATTACK WAS WIPED OUT.")  # 打印"YOUR ATTACK WAS WIPED OUT."
            usr_air = usr_air - num_units  # 用户空军数量减去单位数量
        elif num_units < 2 * usr_air / 3:  # 如果用户单位数量小于用户空军数量的2/3
            print("WE HAD A DOGFIGHT. YOU WON - AND FINISHED YOUR MISSION.")
            # 更新 CPU 的陆军数量为原数量的 2/3
            cpu_army = int(2 * cpu_army / 3)
            # 更新 CPU 的海军数量为原数量的 1/3
            cpu_navy = int(cpu_navy / 3)
            # 更新 CPU 的空军数量为原数量的 1/3
            cpu_air = int(cpu_air / 3)
        else:
            print("YOU WIPED OUT ONE OF MY ARMY PATROLS, BUT I DESTROYED")
            print("TWO NAVY BASES AND BOMBED THREE ARMY BASES.")
            # 更新玩家的陆军数量为原数量的 1/4
            usr_army = int(usr_army / 4)
            # 更新玩家的海军数量为原数量的 1/3
            usr_navy = int(usr_navy / 3)
            # 更新 CPU 的陆军数量为原数量的 2/3
            cpu_army = int(2 * cpu_army / 3)


def attack_second() -> None:
    # 声明使用全局变量
    global usr_army, usr_navy, usr_air, cpu_army, cpu_navy, cpu_air
    global plane_crash_win
    # 初始化单位数量和类型
    num_units = 0
    unit_type = 0

    print()
    print("              YOU           ME")
    print("ARMY           ", end="")  # 打印字符串 "ARMY"，并以空格结尾
    print("%-14s%s\n" % (usr_army, cpu_army), end="")  # 使用格式化字符串打印 usr_army 和 cpu_army 的值，并以换行符结尾
    print("NAVY           ", end="")  # 打印字符串 "NAVY"，并以空格结尾
    print("%-14s%s\n" % (usr_navy, cpu_navy), end="")  # 使用格式化字符串打印 usr_navy 和 cpu_navy 的值，并以换行符结尾
    print("A. F.          ", end="")  # 打印字符串 "A. F."，并以空格结尾
    print("%-14s%s\n" % (usr_air, cpu_air), end="")  # 使用格式化字符串打印 usr_air 和 cpu_air 的值，并以换行符结尾

    while True:  # 进入无限循环
        print("WHAT IS YOUR NEXT MOVE?")  # 打印提示信息
        print("ARMY=1  NAVY=2  AIR FORCE=3")  # 打印提示信息
        print("? ", end="")  # 打印问号，并以空格结尾
        unit_type = int(input())  # 获取用户输入的整数值赋给变量 unit_type
        if not ((unit_type < 1) or (unit_type > 3)):  # 如果 unit_type 不小于 1 且不大于 3，则跳出循环
            break

    while True:  # 进入无限循环
        print("HOW MANY MEN")  # 打印提示信息
        print("? ", end="")  # 打印问号，并以空格结尾
        num_units = int(input())  # 获取用户输入的整数值赋给变量 num_units
        if not (  # 如果 num_units 不小于 0 且不大于 100，则跳出循环
            (num_units < 0)  # 如果单位数量小于0
            or ((unit_type == 1) and (num_units > usr_army))  # 或者（如果单位类型为1且单位数量大于用户陆军数量）
            or ((unit_type == 2) and (num_units > usr_navy))  # 或者（如果单位类型为2且单位数量大于用户海军数量）
            or ((unit_type == 3) and (num_units > usr_air))   # 或者（如果单位类型为3且单位数量大于用户空军数量）
        ):
            break  # 跳出循环

    if unit_type == 1:  # 如果单位类型为1
        if num_units < (cpu_army / 2):  # 如果单位数量小于（CPU陆军数量的一半）
            print("I WIPED OUT YOUR ATTACK!")  # 打印“我消灭了你的进攻！”
            usr_army = usr_army - num_units  # 用户陆军数量减去单位数量
        else:
            print("YOU DESTROYED MY ARMY!")  # 否则打印“你摧毁了我的军队！”
            cpu_army = 0  # CPU陆军数量设为0
    elif unit_type == 2:  # 如果单位类型为2
        if num_units < (cpu_navy / 2):  # 如果单位数量小于（CPU海军数量的一半）
            print("I SUNK TWO OF YOUR BATTLESHIPS, AND MY AIR FORCE")  # 打印“我击沉了你的两艘战舰，还有我的空军”
            print("WIPED OUT YOUR UNGUARDED CAPITOL.")  # 打印“消灭了你不设防的首都。”
            usr_army = int(usr_army / 4)  # 用户陆军数量设为原来的四分之一
            usr_navy = int(usr_navy / 2)  # 用户海军数量设为原来的一半
    else:
        # 如果条件不满足，则打印以下信息
        print("YOUR NAVY SHOT DOWN THREE OF MY XIII PLANES,")
        print("AND SUNK THREE BATTLESHIPS.")
        # 对 CPU 的空军数量进行调整
        cpu_air = int(2 * cpu_air / 3)
        # 对 CPU 的海军数量进行调整
        cpu_navy = int(cpu_navy / 2)
elif unit_type == 3:
    # 如果单位类型为 3，则执行以下代码
    if num_units > (cpu_air / 2):
        # 如果我方单位数量大于 CPU 空军数量的一半，则打印以下信息
        print("MY NAVY AND AIR FORCE IN A COMBINED ATTACK LEFT")
        print("YOUR COUNTRY IN SHAMBLES.")
        # 对我方陆军数量进行调整
        usr_army = int(usr_army / 3)
        # 对我方海军数量进行调整
        usr_navy = int(usr_navy / 3)
        # 对我方空军数量进行调整
        usr_air = int(usr_air / 3)
    else:
        # 如果条件不满足，则打印以下信息
        print("ONE OF YOUR PLANES CRASHED INTO MY HOUSE. I AM DEAD.")
        print("MY COUNTRY FELL APART.")
        # 设置飞机坠毁胜利标志为 True
        plane_crash_win = True

if not plane_crash_win:
    # 如果飞机坠毁胜利标志为 False，则打印以下信息
    print()
    print("FROM THE RESULTS OF BOTH OF YOUR ATTACKS,")
    if plane_crash_win or (
        (usr_army + usr_navy + usr_air) > (int(3 / 2 * (cpu_army + cpu_navy + cpu_air)))
    ):
        # 如果飞机坠毁或者玩家的陆军、海军和空军总和大于对手总和的3/2，则打印胜利信息
        print("YOU WON, OH! SHUCKS!!!!")
    elif (usr_army + usr_navy + usr_air) < int(2 / 3 * (cpu_army + cpu_navy + cpu_air)):
        # 如果玩家的陆军、海军和空军总和小于对手总和的2/3，则打印失败信息
        print("YOU LOST-I CONQUERED YOUR COUNTRY.  IT SERVES YOU")
        print("RIGHT FOR PLAYING THIS STUPID GAME!!!")
    else:
        # 否则打印和平协议信息
        print("THE TREATY OF PARIS CONCLUDED THAT WE TAKE OUR")
        print("RESPECTIVE COUNTRIES AND LIVE IN PEACE.")


def main() -> None:
    # 显示游戏介绍
    show_intro()
    # 获取玩家和对手的军队力量
    get_forces()
    # 进行第一次攻击
    attack_first()
    # 进行第二次攻击
    attack_second()
# 如果当前脚本被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```

这段代码是一个常见的 Python 习惯用法，用于判断当前脚本是否被直接执行。如果当前脚本被直接执行，则会调用 main() 函数。这样做可以使代码更具可重用性，因为可以在其他脚本中导入当前脚本而不会立即执行 main() 函数。
```