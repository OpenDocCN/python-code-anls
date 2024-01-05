# `84_Super_Star_Trek\python\superstartrekins.py`

```
"""
SUPER STARTREK INSTRUCTIONS
MAR 5, 1978

Just the instructions for SUPERSTARTREK

Ported by Dave LeCompte
"""


# 定义一个函数，根据提示信息获取用户输入的是或否，返回布尔值
def get_yes_no(prompt: str) -> bool:
    # 获取用户输入的内容并转换为大写
    response = input(prompt).upper()
    # 返回用户输入的第一个字符是否不是"N"的布尔值
    return response[0] != "N"


# 定义一个函数，打印游戏的标题
def print_header() -> None:
    # 打印12行空行
    for _ in range(12):
        print()
    # 创建一个包含10个空格的字符串
    t10 = " " * 10
    # 打印游戏标题
    print(t10 + "*************************************")
    print(t10 + "*                                   *")  # 打印星号边框
    print(t10 + "*                                   *")  # 打印星号边框
    print(t10 + "*      * * SUPER STAR TREK * *      *")  # 打印游戏名称
    print(t10 + "*                                   *")  # 打印星号边框
    print(t10 + "*                                   *")  # 打印星号边框
    print(t10 + "*************************************")  # 打印星号边框
    for _ in range(8):  # 循环打印空行
        print()


def print_instructions() -> None:
    # 在70年代，用户会被提示打开他们的（打印）TTY以捕获输出到硬拷贝。

    print("      INSTRUCTIONS FOR 'SUPER STAR TREK'")  # 打印游戏说明标题
    print()  # 打印空行
    print("1. WHEN YOU SEE \\COMMAND ?\\ PRINTED, ENTER ONE OF THE LEGAL")  # 打印游戏指令说明
    print("     COMMANDS (NAV,SRS,LRS,PHA,TOR,SHE,DAM,COM, OR XXX).")
    print("2. IF YOU SHOULD TYPE IN AN ILLEGAL COMMAND, YOU'LL GET A SHORT")  # 打印游戏指令说明
    print("     LIST OF THE LEGAL COMMANDS PRINTED OUT.")
    # 打印提示信息，说明某些命令需要输入数据（例如，'NAV' 命令会返回 'COURSE (1-9) ?'）。如果输入非法数据（如负数），则命令将被中止
    print("3. SOME COMMANDS REQUIRE YOU TO ENTER DATA (FOR EXAMPLE, THE")
    print("     'NAV' COMMAND COMES BACK WITH 'COURSE (1-9) ?'.)  IF YOU")
    print("     TYPE IN ILLEGAL DATA (LIKE NEGATIVE NUMBERS), THAN COMMAND")
    print("     WILL BE ABORTED")
    print()
    # 打印提示信息，说明星系被划分为一个 8 x 8 的象限网格，每个象限进一步划分为一个 8 x 8 的扇区网格
    print("     THE GALAXY IS DIVIDED INTO AN 8 X 8 QUADRANT GRID,")
    print("AND EACH QUADRANT IS FURTHER DIVIDED INTO AN 8 X 8 SECTOR GRID.")
    print()
    # 打印提示信息，说明你将被分配一个起始点在星系中的某处，开始担任星舰“企业号”的指挥官；你的任务是寻找并摧毁威胁联合星际国的克林贡战舰舰队
    print("     YOU WILL BE ASSIGNED A STARTING POINT SOMEWHERE IN THE")
    print("GALAXY TO BEGIN A TOUR OF DUTY AS COMANDER OF THE STARSHIP")
    print("\\ENTERPRISE\\; YOUR MISSION: TO SEEK AND DESTROY THE FLEET OF")
    print("KLINGON WARWHIPS WHICH ARE MENACING THE UNITED FEDERATION OF")
    print("PLANETS.")
    print()
    # 打印提示信息，说明作为“企业号”舰长，你有以下命令可用：
    print("     YOU HAVE THE FOLLOWING COMMANDS AVAILABLE TO YOU AS CAPTAIN")
    print("OF THE STARSHIP ENTERPRISE:")
    print()
    # 打印提示信息，说明“NAV”命令 = 曲速引擎控制 -- 航向以循环数字方式表示，如下所示
    print("\\NAV\\ COMMAND = WARP ENGINE CONTROL --")
    print("     COURSE IS IN A CIRCULAR NUMERICAL      4  3  2")
    print("     VECTOR ARRANGEMENT AS SHOWN             . . .")
    print("     INTEGER AND REAL VALUES MAY BE           ...")  # 打印整数和实数值可能被使用
    print("     USED.  (THUS COURSE 1.5 IS HALF-     5 ---*--- 1")  # 打印使用（因此，课程1.5是1和2之间的一半）
    print("     WAY BETWEEN 1 AND 2                      ...")  # 打印在1和2之间的方式
    print("                                             . . .")  # 打印空格
    print("     VALUES MAY APPROACH 9.0, WHICH         6  7  8")  # 打印值可能接近9.0，这本身相当于1.0
    print("     ITSELF IS EQUIVALENT TO 1.0")  # 打印它本身等同于1.0
    print("                                            COURSE")  # 打印课程
    print("     ONE WARP FACTOR IS THE SIZE OF ")  # 打印一个超空间因子是一个象限的大小
    print("     ONE QUADTANT.  THEREFORE, TO GET")  # 打印一个象限。因此，要到达从象限6,5到5,5，你将使用课程3，超空间因子1。
    print("     FROM QUADRANT 6,5 TO 5,5, YOU WOULD")
    print("     USE COURSE 3, WARP FACTOR 1.")
    print()
    print("\\SRS\\ COMMAND = SHORT RANGE SENSOR SCAN")  # 打印\SRS\ 命令 = 短程传感器扫描
    print("     SHOWS YOU A SCAN OF YOUR PRESENT QUADRANT.")  # 打印显示你当前象限的扫描。
    print()
    print("     SYMBOLOGY ON YOUR SENSOR SCREEN IS AS FOLLOWS:")  # 打印你传感器屏幕上的符号是如下：
    print("        <*> = YOUR STARSHIP'S POSITION")  # 打印<*> = 你的星舰位置
    print("        +K+ = KLINGON BATTLE CRUISER")  # 打印+K+ = 克林贡战舰
    print("        >!< = FEDERATION STARBASE (REFUEL/REPAIR/RE-ARM HERE!)")  # 打印>!< = 联邦星舰基地（在这里加油/修理/重新武装！）
    print("         *  = STAR")  # 打印* = 星星
    print()  # 打印空行
    print("     A CONDENSED 'STATUS REPORT' WILL ALSO BE PRESENTED.")  # 打印状态报告的简要信息
    print()  # 打印空行
    print("\\LRS\\ COMMAND = LONG RANGE SENSOR SCAN")  # 打印关于长程传感器扫描的命令说明
    print("     SHOWS CONDITIONS IN SPACE FOR ONE QUADRANT ON EACH SIDE")  # 打印长程传感器扫描的作用说明
    print("     OF THE ENTERPRISE (WHICH IS IN THE MIDDLE OF THE SCAN)")  # 打印长程传感器扫描的作用说明
    print("     THE SCAN IS CODED IN THE FORM \\###\\, WHERE TH UNITS DIGIT")  # 打印长程传感器扫描的格式说明
    print("     IS THE NUMBER OF STARS, THE TENS DIGIT IS THE NUMBER OF")  # 打印长程传感器扫描的格式说明
    print("     STARBASES, AND THE HUNDRESDS DIGIT IS THE NUMBER OF")  # 打印长程传感器扫描的格式说明
    print("     KLINGONS.")  # 打印长程传感器扫描的格式说明
    print()  # 打印空行
    print("     EXAMPLE - 207 = 2 KLINGONS, NO STARBASES, & 7 STARS.")  # 打印长程传感器扫描的示例
    print()  # 打印空行
    print("\\PHA\\ COMMAND = PHASER CONTROL.")  # 打印关于相位控制的命令说明
    print("     ALLOWS YOU TO DESTROY THE KLINGON BATTLE CRUISERS BY ")  # 打印相位控制的作用说明
    print("     ZAPPING THEM WITH SUITABLY LARGE UNITS OF ENERGY TO")  # 打印相位控制的作用说明
    print("     DEPLETE THEIR SHIELD POWER.  (REMEMBER, KLINGONS HAVE")  # 打印相位控制的作用说明
    print("     PHASERS TOO!)")  # 打印相位控制的作用说明
    print()  # 打印空行
    print("\\TOR\\ COMMAND = PHOTON TORPEDO CONTROL")  # 打印关于光子鱼雷控制的命令说明
    # 打印游戏提示信息
    print("     TORPEDO COURSE IS THE SAME AS USED IN WARP ENGINE CONTROL")
    print("     IF YOU HIT THE KLINGON VESSEL, HE IS DESTROYED AND")
    print("     CANNOT FIRE BACK AT YOU.  IF YOU MISS, YOU ARE SUBJECT TO")
    print("     HIS PHASER FIRE.  IN EITHER CASE, YOU ARE ALSO SUBJECT TO ")
    print("     THE PHASER FIRE OF ALL OTHER KLINGONS IN THE QUADRANT.")
    print()
    print("     THE LIBRARY-COMPUTER (\\COM\\ COMMAND) HAS AN OPTION TO ")
    print("     COMPUTE TORPEDO TRAJECTORY FOR YOU (OPTION 2)")
    print()
    # 打印游戏指令信息
    print("\\SHE\\ COMMAND = SHIELD CONTROL")
    print("     DEFINES THE NUMBER OF ENERGY UNITS TO BE ASSIGNED TO THE")
    print("     SHIELDS.  ENERGY IS TAKEN FROM TOTAL SHIP'S ENERGY.  NOTE")
    print("     THAN THE STATUS DISPLAY TOTAL ENERGY INCLUDES SHIELD ENERGY")
    print()
    print("\\DAM\\ COMMAND = DAMMAGE CONTROL REPORT")
    print("     GIVES THE STATE OF REPAIR OF ALL DEVICES.  WHERE A NEGATIVE")
    print("     'STATE OF REPAIR' SHOWS THAT THE DEVICE IS TEMPORARILY")
    print("     DAMAGED.")
    print()
    # 打印游戏指令信息
    print("\\COM\\ COMMAND = LIBRARY-COMPUTER")
    # 打印出计算机库存储的六个选项
    print("     THE LIBRARY-COMPUTER CONTAINS SIX OPTIONS:")
    # 打印出选项0的说明
    print("     OPTION 0 = CUMULATIVE GALACTIC RECORD")
    # 打印出选项1的说明
    print("        THIS OPTION SHOWES COMPUTER MEMORY OF THE RESULTS OF ALL")
    print("        PREVIOUS SHORT AND LONG RANGE SENSOR SCANS")
    # 打印出选项2的说明
    print("     OPTION 1 = STATUS REPORT")
    print("        THIS OPTION SHOWS THE NUMBER OF KLINGONS, STARDATES,")
    print("        AND STARBASES REMAINING IN THE GAME.")
    # 打印出选项3的说明
    print("     OPTION 2 = PHOTON TORPEDO DATA")
    print("        WHICH GIVES DIRECTIONS AND DISTANCE FROM THE ENTERPRISE")
    print("        TO ALL KLINGONS IN YOUR QUADRANT")
    # 打印出选项4的说明
    print("     OPTION 3 = STARBASE NAV DATA")
    print("        THIS OPTION GIVES DIRECTION AND DISTANCE TO ANY ")
    print("        STARBASE WITHIN YOUR QUADRANT")
    # 打印出选项5的说明
    print("     OPTION 4 = DIRECTION/DISTANCE CALCULATOR")
    print("        THIS OPTION ALLOWS YOU TO ENTER COORDINATES FOR")
    print("        DIRECTION/DISTANCE CALCULATIONS")
    # 打印出选项6的说明
    print("     OPTION 5 = GALACTIC /REGION NAME/ MAP")
    print("        THIS OPTION PRINTS THE NAMES OF THE SIXTEEN MAJOR ")
    print("        GALACTIC REGIONS REFERRED TO IN THE GAME.")
# 定义主函数
def main() -> None:
    # 打印标题
    print_header()
    # 如果用户不需要说明，则返回
    if not get_yes_no("DO YOU NEED INSTRUCTIONS (Y/N)? "):
        return
    # 打印说明
    print_instructions()

# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()
```