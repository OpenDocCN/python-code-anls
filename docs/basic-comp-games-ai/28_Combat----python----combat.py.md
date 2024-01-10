# `basic-computer-games\28_Combat\python\combat.py`

```
# 设置最大单位数
MAX_UNITS = 72000
# 初始化飞机坠毁标志
plane_crash_win = False
# 初始化用户陆军数量
usr_army = 0
# 初始化用户海军数量
usr_navy = 0
# 初始化用户空军数量
usr_air = 0
# 初始化CPU陆军数量
cpu_army = 30000
# 初始化CPU海军数量
cpu_navy = 20000
# 初始化CPU空军数量
cpu_air = 22000

# 显示游戏介绍
def show_intro() -> None:
    global MAX_UNITS
    # 打印游戏标题
    print(" " * 32 + "COMBAT")
    print(" " * 14 + "CREATIVE COMPUTING MORRISTOWN, NEW JERSEY")
    print("\n\n")
    print("I AM AT WAR WITH YOU.")
    print("WE HAVE " + str(MAX_UNITS) + " SOLDIERS APIECE.")

# 获取用户军队力量
def get_forces() -> None:
    global usr_army, usr_navy, usr_air
    # 循环直到用户分配的军队力量不超过最大单位数
    while True:
        print("DISTRIBUTE YOUR FORCES.")
        print("              ME              YOU")
        print("ARMY           " + str(cpu_army) + "        ? ", end="")
        usr_army = int(input())
        print("NAVY           " + str(cpu_navy) + "        ? ", end="")
        usr_navy = int(input())
        print("A. F.          " + str(cpu_air) + "        ? ", end="")
        usr_air = int(input())
        if (usr_army + usr_navy + usr_air) <= MAX_UNITS:
            break

# 用户先攻击
def attack_first() -> None:
    global usr_army, usr_navy, usr_air
    global cpu_army, cpu_navy, cpu_air
    num_units = 0
    unit_type = 0
    # 循环直到用户选择有效的军种类型
    while True:
        print("YOU ATTACK FIRST. TYPE (1) FOR ARMY; (2) FOR NAVY;")
        print("AND (3) FOR AIR FORCE.")
        print("?", end=" ")
        unit_type = int(input())
        if not (unit_type < 1 or unit_type > 3):
            break
    # 循环直到用户输入有效的军队数量
    while True:
        print("HOW MANY MEN")
        print("?", end=" ")
        num_units = int(input())
        if not (
            (num_units < 0)
            or ((unit_type == 1) and (num_units > usr_army))
            or ((unit_type == 2) and (num_units > usr_navy))
            or ((unit_type == 3) and (num_units > usr_air))
        ):
            break
    # 如果单位类型为1
    if unit_type == 1:
        # 如果攻击单位数量小于玩家军队的三分之一
        if num_units < (usr_army / 3):
            # 打印提示信息，减少玩家军队数量
            print("YOU LOST " + str(num_units) + " MEN FROM YOUR ARMY.")
            usr_army = usr_army - num_units
        # 如果攻击单位数量小于玩家军队的三分之二
        elif num_units < (2 * usr_army / 3):
            # 打印提示信息，减少玩家军队数量，清空对手军队
            print(
                "YOU LOST "
                + str(int(num_units / 3))
                + " MEN, BUT I LOST "
                + str(int(2 * cpu_army / 3))
            )
            usr_army = int(usr_army - (num_units / 3))
            cpu_army = 0
        # 否则
        else:
            # 打印提示信息，减少玩家军队数量和空军数量，减少对手海军数量
            print("YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO")
            print("OF YOUR AIR FORCE BASES AND 3 ARMY BASES.")
            usr_army = int(usr_army / 3)
            usr_air = int(usr_air / 3)
            cpu_navy = int(2 * cpu_navy / 3)
    # 如果单位类型为2
    elif unit_type == 2:
        # 如果攻击单位数量小于对手海军的三分之一
        if num_units < cpu_navy / 3:
            # 打印提示信息，减少玩家海军数量
            print("YOUR ATTACK WAS STOPPED!")
            usr_navy = usr_navy - num_units
        # 如果攻击单位数量小于对手海军的三分之二
        elif num_units < 2 * cpu_navy / 3:
            # 打印提示信息，减少对手海军数量
            print("YOU DESTROYED " + str(int(2 * cpu_navy / 3)) + " OF MY ARMY.")
            cpu_navy = int(cpu_navy / 3)
        # 否则
        else:
            # 打印提示信息，减少玩家军队数量和空军数量，减少对手海军数量
            print("YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO")
            print("OF YOUR AIR FORCE BASES AND 3 ARMY BASES.")
            usr_army = int(usr_army / 3)
            usr_air = int(usr_air / 3)
            cpu_navy = int(2 * cpu_navy / 3)
    # 如果单位类型为3
    elif unit_type == 3:
        # 如果敌方单位数量小于用户空军数量的三分之一
        if num_units < usr_air / 3:
            # 打印信息：你的攻击被消灭了
            print("YOUR ATTACK WAS WIPED OUT.")
            # 减去相应数量的用户空军
            usr_air = usr_air - num_units
        # 如果敌方单位数量小于用户空军数量的三分之二
        elif num_units < 2 * usr_air / 3:
            # 打印信息：我们进行了空战，你赢了，并完成了你的任务
            print("WE HAD A DOGFIGHT. YOU WON - AND FINISHED YOUR MISSION.")
            # 更新敌方军队、海军和空军数量
            cpu_army = int(2 * cpu_army / 3)
            cpu_navy = int(cpu_navy / 3)
            cpu_air = int(cpu_air / 3)
        # 如果以上条件都不满足
        else:
            # 打印信息：你消灭了我的一个军队巡逻队，但我摧毁了两个海军基地并轰炸了三个陆军基地
            print("YOU WIPED OUT ONE OF MY ARMY PATROLS, BUT I DESTROYED")
            print("TWO NAVY BASES AND BOMBED THREE ARMY BASES.")
            # 更新用户和敌方军队、海军数量
            usr_army = int(usr_army / 4)
            usr_navy = int(usr_navy / 3)
            cpu_army = int(2 * cpu_army / 3)
# 定义一个名为 attack_second 的函数，无返回值
def attack_second() -> None:
    # 声明全局变量
    global usr_army, usr_navy, usr_air, cpu_army, cpu_navy, cpu_air
    global plane_crash_win
    # 初始化变量
    num_units = 0
    unit_type = 0

    # 打印空行
    print()
    # 打印表头
    print("              YOU           ME")
    # 打印军队数量
    print("ARMY           ", end="")
    print("%-14s%s\n" % (usr_army, cpu_army), end="")
    # 打印海军数量
    print("NAVY           ", end="")
    print("%-14s%s\n" % (usr_navy, cpu_navy), end="")
    # 打印空军数量
    print("A. F.          ", end="")
    print("%-14s%s\n" % (usr_air, cpu_air), end="")

    # 循环直到用户输入合法的单位类型
    while True:
        print("WHAT IS YOUR NEXT MOVE?")
        print("ARMY=1  NAVY=2  AIR FORCE=3")
        print("? ", end="")
        unit_type = int(input())
        if not ((unit_type < 1) or (unit_type > 3)):
            break

    # 循环直到用户输入合法的单位数量
    while True:
        print("HOW MANY MEN")
        print("? ", end="")
        num_units = int(input())
        if not (
            (num_units < 0)
            or ((unit_type == 1) and (num_units > usr_army))
            or ((unit_type == 2) and (num_units > usr_navy))
            or ((unit_type == 3) and (num_units > usr_air))
        ):
            break

    # 根据用户选择的单位类型和数量进行不同的处理
    if unit_type == 1:
        if num_units < (cpu_army / 2):
            print("I WIPED OUT YOUR ATTACK!")
            usr_army = usr_army - num_units
        else:
            print("YOU DESTROYED MY ARMY!")
            cpu_army = 0
    elif unit_type == 2:
        if num_units < (cpu_navy / 2):
            print("I SUNK TWO OF YOUR BATTLESHIPS, AND MY AIR FORCE")
            print("WIPED OUT YOUR UNGUARDED CAPITOL.")
            usr_army = int(usr_army / 4)
            usr_navy = int(usr_navy / 2)
        else:
            print("YOUR NAVY SHOT DOWN THREE OF MY XIII PLANES,")
            print("AND SUNK THREE BATTLESHIPS.")
            cpu_air = int(2 * cpu_air / 3)
            cpu_navy = int(cpu_navy / 2)
    # 如果单位类型为3
    elif unit_type == 3:
        # 如果玩家单位数量大于CPU空军数量的一半
        if num_units > (cpu_air / 2):
            # 打印消息
            print("MY NAVY AND AIR FORCE IN A COMBINED ATTACK LEFT")
            print("YOUR COUNTRY IN SHAMBLES.")
            # 玩家陆军、海军、空军数量各减少三分之一
            usr_army = int(usr_army / 3)
            usr_navy = int(usr_navy / 3)
            usr_air = int(usr_air / 3)
        else:
            # 打印消息
            print("ONE OF YOUR PLANES CRASHED INTO MY HOUSE. I AM DEAD.")
            print("MY COUNTRY FELL APART.")
            # 设置飞机坠毁胜利标志为True
            plane_crash_win = True

    # 如果飞机坠毁胜利标志为False
    if not plane_crash_win:
        # 打印空行和消息
        print()
        print("FROM THE RESULTS OF BOTH OF YOUR ATTACKS,")

    # 如果飞机坠毁胜利标志为True或者玩家单位总数大于CPU单位总数的3/2
    if plane_crash_win or (
        (usr_army + usr_navy + usr_air) > (int(3 / 2 * (cpu_army + cpu_navy + cpu_air)))
    ):
        # 打印消息
        print("YOU WON, OH! SHUCKS!!!!")
    # 如果玩家单位总数小于CPU单位总数的2/3
    elif (usr_army + usr_navy + usr_air) < int(2 / 3 * (cpu_army + cpu_navy + cpu_air)):
        # 打印消息
        print("YOU LOST-I CONQUERED YOUR COUNTRY.  IT SERVES YOU")
        print("RIGHT FOR PLAYING THIS STUPID GAME!!!")
    # 其他情况
    else:
        # 打印消息
        print("THE TREATY OF PARIS CONCLUDED THAT WE TAKE OUR")
        print("RESPECTIVE COUNTRIES AND LIVE IN PEACE.")
# 定义主函数，没有返回值
def main() -> None:
    # 显示游戏介绍
    show_intro()
    # 获取玩家和敌人的力量值
    get_forces()
    # 进行第一次攻击
    attack_first()
    # 进行第二次攻击

# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```