# `basic-computer-games\84_Super_Star_Trek\python\superstartrekins.py`

```py
"""
SUPER STARTREK INSTRUCTIONS
MAR 5, 1978

Just the instructions for SUPERSTARTREK

Ported by Dave LeCompte
"""

# 定义一个函数，用于获取用户输入的是“是”还是“否”
def get_yes_no(prompt: str) -> bool:
    # 获取用户输入并转换为大写
    response = input(prompt).upper()
    # 返回用户输入的第一个字符是否不是“N”
    return response[0] != "N"

# 定义一个函数，用于打印游戏的标题
def print_header() -> None:
    # 打印空行
    for _ in range(12):
        print()
    t10 = " " * 10
    # 打印游戏标题
    print(t10 + "*************************************")
    print(t10 + "*                                   *")
    print(t10 + "*                                   *")
    print(t10 + "*      * * SUPER STAR TREK * *      *")
    print(t10 + "*                                   *")
    print(t10 + "*                                   *")
    print(t10 + "*************************************")
    # 打印空行
    for _ in range(8):
        print()

# 定义一个函数，用于打印游戏的说明
def print_instructions() -> None:
    # 在70年代，用户会被提示打开他们的（打印）TTY，以便捕获输出到硬拷贝。
    print("      INSTRUCTIONS FOR 'SUPER STAR TREK'")
    print()
    print("1. WHEN YOU SEE \\COMMAND ?\\ PRINTED, ENTER ONE OF THE LEGAL")
    print("     COMMANDS (NAV,SRS,LRS,PHA,TOR,SHE,DAM,COM, OR XXX).")
    print("2. IF YOU SHOULD TYPE IN AN ILLEGAL COMMAND, YOU'LL GET A SHORT")
    print("     LIST OF THE LEGAL COMMANDS PRINTED OUT.")
    print("3. SOME COMMANDS REQUIRE YOU TO ENTER DATA (FOR EXAMPLE, THE")
    print("     'NAV' COMMAND COMES BACK WITH 'COURSE (1-9) ?'.)  IF YOU")
    print("     TYPE IN ILLEGAL DATA (LIKE NEGATIVE NUMBERS), THAN COMMAND")
    print("     WILL BE ABORTED")
    print()
    print("     THE GALAXY IS DIVIDED INTO AN 8 X 8 QUADRANT GRID,")
    print("AND EACH QUADRANT IS FURTHER DIVIDED INTO AN 8 X 8 SECTOR GRID.")
    print()
    print("     YOU WILL BE ASSIGNED A STARTING POINT SOMEWHERE IN THE")
    print("GALAXY TO BEGIN A TOUR OF DUTY AS COMANDER OF THE STARSHIP")
    print("\\ENTERPRISE\\; YOUR MISSION: TO SEEK AND DESTROY THE FLEET OF")
    print("KLINGON WARWHIPS WHICH ARE MENACING THE UNITED FEDERATION OF")
    # 打印"PLANETS."
    print("PLANETS.")
    # 打印空行
    print()
    # 打印"     YOU HAVE THE FOLLOWING COMMANDS AVAILABLE TO YOU AS CAPTAIN"
    print("     YOU HAVE THE FOLLOWING COMMANDS AVAILABLE TO YOU AS CAPTAIN")
    # 打印"OF THE STARSHIP ENTERPRISE:"
    print("OF THE STARSHIP ENTERPRISE:")
    # 打印空行
    print()
    # 打印"\NAV\ COMMAND = WARP ENGINE CONTROL --"
    print("\\NAV\\ COMMAND = WARP ENGINE CONTROL --")
    # 打印"     COURSE IS IN A CIRCULAR NUMERICAL      4  3  2"
    print("     COURSE IS IN A CIRCULAR NUMERICAL      4  3  2")
    # 打印"     VECTOR ARRANGEMENT AS SHOWN             . . ."
    print("     VECTOR ARRANGEMENT AS SHOWN             . . .")
    # 打印"     INTEGER AND REAL VALUES MAY BE           ..."
    print("     INTEGER AND REAL VALUES MAY BE           ...")
    # 打印"     USED.  (THUS COURSE 1.5 IS HALF-     5 ---*--- 1"
    print("     USED.  (THUS COURSE 1.5 IS HALF-     5 ---*--- 1")
    # 打印"     WAY BETWEEN 1 AND 2                      ..."
    print("     WAY BETWEEN 1 AND 2                      ...")
    # 打印"                                             . . ."
    print("                                             . . .")
    # 打印"     VALUES MAY APPROACH 9.0, WHICH         6  7  8"
    print("     VALUES MAY APPROACH 9.0, WHICH         6  7  8")
    # 打印"     ITSELF IS EQUIVALENT TO 1.0"
    print("     ITSELF IS EQUIVALENT TO 1.0")
    # 打印"                                            COURSE"
    print("                                            COURSE")
    # 打印"     ONE WARP FACTOR IS THE SIZE OF "
    print("     ONE WARP FACTOR IS THE SIZE OF ")
    # 打印"     ONE QUADTANT.  THEREFORE, TO GET"
    print("     ONE QUADTANT.  THEREFORE, TO GET")
    # 打印"     FROM QUADRANT 6,5 TO 5,5, YOU WOULD"
    print("     FROM QUADRANT 6,5 TO 5,5, YOU WOULD")
    # 打印"     USE COURSE 3, WARP FACTOR 1."
    print("     USE COURSE 3, WARP FACTOR 1.")
    # 打印空行
    print()
    # 打印"\SRS\ COMMAND = SHORT RANGE SENSOR SCAN"
    print("\\SRS\\ COMMAND = SHORT RANGE SENSOR SCAN")
    # 打印"     SHOWS YOU A SCAN OF YOUR PRESENT QUADRANT."
    print("     SHOWS YOU A SCAN OF YOUR PRESENT QUADRANT.")
    # 打印空行
    print()
    # 打印"     SYMBOLOGY ON YOUR SENSOR SCREEN IS AS FOLLOWS:"
    print("     SYMBOLOGY ON YOUR SENSOR SCREEN IS AS FOLLOWS:")
    # 打印"        <*> = YOUR STARSHIP'S POSITION"
    print("        <*> = YOUR STARSHIP'S POSITION")
    # 打印"        +K+ = KLINGON BATTLE CRUISER"
    print("        +K+ = KLINGON BATTLE CRUISER")
    # 打印"        >!< = FEDERATION STARBASE (REFUEL/REPAIR/RE-ARM HERE!)"
    print("        >!< = FEDERATION STARBASE (REFUEL/REPAIR/RE-ARM HERE!)")
    # 打印"         *  = STAR"
    print("         *  = STAR")
    # 打印空行
    print()
    # 打印"     A CONDENSED 'STATUS REPORT' WILL ALSO BE PRESENTED."
    print("     A CONDENSED 'STATUS REPORT' WILL ALSO BE PRESENTED.")
    # 打印空行
    print()
    # 打印"\LRS\ COMMAND = LONG RANGE SENSOR SCAN"
    print("\\LRS\\ COMMAND = LONG RANGE SENSOR SCAN")
    # 打印"     SHOWS CONDITIONS IN SPACE FOR ONE QUADRANT ON EACH SIDE"
    print("     SHOWS CONDITIONS IN SPACE FOR ONE QUADRANT ON EACH SIDE")
    # 打印"     OF THE ENTERPRISE (WHICH IS IN THE MIDDLE OF THE SCAN)"
    print("     OF THE ENTERPRISE (WHICH IS IN THE MIDDLE OF THE SCAN)")
    # 打印"     THE SCAN IS CODED IN THE FORM \\###\\, WHERE TH UNITS DIGIT"
    print("     THE SCAN IS CODED IN THE FORM \\###\\, WHERE TH UNITS DIGIT")
    # 打印"     IS THE NUMBER OF STARS, THE TENS DIGIT IS THE NUMBER OF"
    print("     IS THE NUMBER OF STARS, THE TENS DIGIT IS THE NUMBER OF")
    # 打印"     STARBASES, AND THE HUNDRESDS DIGIT IS THE NUMBER OF"
    print("     STARBASES, AND THE HUNDRESDS DIGIT IS THE NUMBER OF")
    # 打印"     KLINGONS."
    print("     KLINGONS.")
    # 打印空行
    print()
    # 打印游戏示例信息
    print("     EXAMPLE - 207 = 2 KLINGONS, NO STARBASES, & 7 STARS.")
    # 打印空行
    print()
    # 打印关于 PHA 命令的说明
    print("\\PHA\\ COMMAND = PHASER CONTROL.")
    print("     ALLOWS YOU TO DESTROY THE KLINGON BATTLE CRUISERS BY ")
    print("     ZAPPING THEM WITH SUITABLY LARGE UNITS OF ENERGY TO")
    print("     DEPLETE THEIR SHIELD POWER.  (REMEMBER, KLINGONS HAVE")
    print("     PHASERS TOO!)")
    # 打印空行
    print()
    # 打印关于 TOR 命令的说明
    print("\\TOR\\ COMMAND = PHOTON TORPEDO CONTROL")
    print("     TORPEDO COURSE IS THE SAME AS USED IN WARP ENGINE CONTROL")
    print("     IF YOU HIT THE KLINGON VESSEL, HE IS DESTROYED AND")
    print("     CANNOT FIRE BACK AT YOU.  IF YOU MISS, YOU ARE SUBJECT TO")
    print("     HIS PHASER FIRE.  IN EITHER CASE, YOU ARE ALSO SUBJECT TO ")
    print("     THE PHASER FIRE OF ALL OTHER KLINGONS IN THE QUADRANT.")
    # 打印空行
    print()
    # 打印关于 SHE 命令的说明
    print("\\SHE\\ COMMAND = SHIELD CONTROL")
    print("     DEFINES THE NUMBER OF ENERGY UNITS TO BE ASSIGNED TO THE")
    print("     SHIELDS.  ENERGY IS TAKEN FROM TOTAL SHIP'S ENERGY.  NOTE")
    print("     THAN THE STATUS DISPLAY TOTAL ENERGY INCLUDES SHIELD ENERGY")
    # 打印空行
    print()
    # 打印关于 DAM 命令的说明
    print("\\DAM\\ COMMAND = DAMMAGE CONTROL REPORT")
    print("     GIVES THE STATE OF REPAIR OF ALL DEVICES.  WHERE A NEGATIVE")
    print("     'STATE OF REPAIR' SHOWS THAT THE DEVICE IS TEMPORARILY")
    print("     DAMAGED.")
    # 打印空行
    print()
    # 打印关于 COM 命令的说明
    print("\\COM\\ COMMAND = LIBRARY-COMPUTER")
    print("     THE LIBRARY-COMPUTER CONTAINS SIX OPTIONS:")
    print("     OPTION 0 = CUMULATIVE GALACTIC RECORD")
    print("        THIS OPTION SHOWES COMPUTER MEMORY OF THE RESULTS OF ALL")
    print("        PREVIOUS SHORT AND LONG RANGE SENSOR SCANS")
    print("     OPTION 1 = STATUS REPORT")
    print("        THIS OPTION SHOWS THE NUMBER OF KLINGONS, STARDATES,")
    print("        AND STARBASES REMAINING IN THE GAME.")
    # 打印选项2的说明
    print("     OPTION 2 = PHOTON TORPEDO DATA")
    # 打印选项2的详细说明
    print("        WHICH GIVES DIRECTIONS AND DISTANCE FROM THE ENTERPRISE")
    print("        TO ALL KLINGONS IN YOUR QUADRANT")
    # 打印选项3的说明
    print("     OPTION 3 = STARBASE NAV DATA")
    # 打印选项3的详细说明
    print("        THIS OPTION GIVES DIRECTION AND DISTANCE TO ANY ")
    print("        STARBASE WITHIN YOUR QUADRANT")
    # 打印选项4的说明
    print("     OPTION 4 = DIRECTION/DISTANCE CALCULATOR")
    # 打印选项4的详细说明
    print("        THIS OPTION ALLOWS YOU TO ENTER COORDINATES FOR")
    print("        DIRECTION/DISTANCE CALCULATIONS")
    # 打印选项5的说明
    print("     OPTION 5 = GALACTIC /REGION NAME/ MAP")
    # 打印选项5的详细说明
    print("        THIS OPTION PRINTS THE NAMES OF THE SIXTEEN MAJOR ")
    print("        GALACTIC REGIONS REFERRED TO IN THE GAME.")
# 定义主函数，没有返回值
def main() -> None:
    # 打印标题
    print_header()
    # 如果用户不需要说明，则直接返回
    if not get_yes_no("DO YOU NEED INSTRUCTIONS (Y/N)? "):
        return
    # 打印说明
    print_instructions()

# 如果当前脚本作为主程序执行，则调用主函数
if __name__ == "__main__":
    main()
```