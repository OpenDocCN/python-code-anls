# `basic-computer-games\28_Combat\python\combat.py`

```

# 最大单位数
MAX_UNITS = 72000
# 飞机坠毁胜利标志
plane_crash_win = False
# 用户陆军
usr_army = 0
# 用户海军
usr_navy = 0
# 用户空军
usr_air = 0
# CPU陆军
cpu_army = 30000
# CPU海军
cpu_navy = 20000
# CPU空军
cpu_air = 22000

# 显示游戏介绍
def show_intro() -> None:
    global MAX_UNITS

    print(" " * 32 + "COMBAT")
    print(" " * 14 + "CREATIVE COMPUTING MORRISTOWN, NEW JERSEY")
    print("\n\n")
    print("I AM AT WAR WITH YOU.")
    print("WE HAVE " + str(MAX_UNITS) + " SOLDIERS APIECE.")

# 获取用户的军队力量分配
def get_forces() -> None:
    global usr_army, usr_navy, usr_air

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

    while True:
        print("YOU ATTACK FIRST. TYPE (1) FOR ARMY; (2) FOR NAVY;")
        print("AND (3) FOR AIR FORCE.")
        print("?", end=" ")
        unit_type = int(input())
        if not (unit_type < 1 or unit_type > 3):
            break

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

    # 根据不同的军种和数量进行不同的攻击结果处理
    # ...

# 用户后攻击
def attack_second() -> None:
    global usr_army, usr_navy, usr_air, cpu_army, cpu_navy, cpu_air
    global plane_crash_win
    num_units = 0
    unit_type = 0

    # 显示双方军队力量
    # ...

    while True:
        print("WHAT IS YOUR NEXT MOVE?")
        print("ARMY=1  NAVY=2  AIR FORCE=3")
        print("? ", end="")
        unit_type = int(input())
        if not ((unit_type < 1) or (unit_type > 3)):
            break

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

    # 根据不同的军种和数量进行不同的攻击结果处理
    # ...

# 主函数
def main() -> None:
    show_intro()
    get_forces()
    attack_first()
    attack_second()

# 程序入口
if __name__ == "__main__":
    main()

```