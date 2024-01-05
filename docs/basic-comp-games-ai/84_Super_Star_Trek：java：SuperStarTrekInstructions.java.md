# `84_Super_Star_Trek\java\SuperStarTrekInstructions.java`

```
import java.io.BufferedReader; // 导入用于读取输入流的 BufferedReader 类
import java.io.IOException; // 导入处理输入输出异常的 IOException 类
import java.io.InputStreamReader; // 导入用于读取输入流的 InputStreamReader 类
import java.util.stream.Collectors; // 导入用于收集流数据的 Collectors 类
import java.util.stream.IntStream; // 导入用于处理整数流的 IntStream 类

/**
 * SUPER STARTREK INSTRUCTIONS
 * MAR 5, 1978
 * Just the instructions for SUPERSTARTREK
 *
 * Ported to Java in Jan-Feb 2022 by
 * Taciano Dreckmann Perez (taciano.perez@gmail.com)
 */
public class SuperStarTrekInstructions {

    public static void main(String[] args) {
        printBanner(); // 调用打印横幅的方法

        final String reply = inputStr("DO YOU NEED INSTRUCTIONS (Y/N)? "); // 获取用户输入的字符串

        if ("Y".equals(reply)) { // 如果用户输入的字符串为 "Y"
            printInstructions();  # 调用printInstructions()函数，打印游戏的说明和指导
        }
    }

    static void printBanner() {
        print(tab(10)+"*************************************");  # 打印游戏的横幅
        print(tab(10)+"*                                   *");
        print(tab(10)+"*                                   *");
        print(tab(10)+"*      * * SUPER STAR TREK * *      *");  # 打印游戏名称
        print(tab(10)+"*                                   *");
        print(tab(10)+"*                                   *");
        print(tab(10)+"*************************************");
    }

    static void printInstructions() {
        print("      INSTRUCTIONS FOR 'SUPER STAR TREK'");  # 打印游戏的说明
        print("");
        print("1. WHEN YOU SEE \\COMMAND ?\\ PRINTED, ENTER ONE OF THE LEGAL");  # 打印游戏指令的说明
        print("     COMMANDS (NAV,SRS,LRS,PHA,TOR,SHE,DAM,COM, OR XXX).");
        print("2. IF YOU SHOULD TYPE IN AN ILLEGAL COMMAND, YOU'LL GET A SHORT");  # 打印游戏指令的说明
        # 打印合法命令列表
        print("     LIST OF THE LEGAL COMMANDS PRINTED OUT.");
        # 打印提示信息，说明有些命令需要输入数据，如果输入非法数据则命令将被中止
        print("3. SOME COMMANDS REQUIRE YOU TO ENTER DATA (FOR EXAMPLE, THE");
        print("     'NAV' COMMAND COMES BACK WITH 'COURSE (1-9) ?'.)  IF YOU");
        print("     TYPE IN ILLEGAL DATA (LIKE NEGATIVE NUMBERS), THAN COMMAND");
        print("     WILL BE ABORTED");
        print("");
        # 打印关于星系的网格划分信息
        print("     THE GALAXY IS DIVIDED INTO AN 8 X 8 QUADRANT GRID,");
        print("AND EACH QUADRANT IS FURTHER DIVIDED INTO AN 8 X 8 SECTOR GRID.");
        print("");
        # 打印关于任务的起始信息
        print("     YOU WILL BE ASSIGNED A STARTING POINT SOMEWHERE IN THE");
        print("GALAXY TO BEGIN A TOUR OF DUTY AS COMANDER OF THE STARSHIP");
        print("\\ENTERPRISE\\; YOUR MISSION: TO SEEK AND DESTROY THE FLEET OF");
        print("KLINGON WARWHIPS WHICH ARE MENACING THE UNITED FEDERATION OF");
        print("PLANETS.");
        print("");
        # 打印可用的命令列表
        print("     YOU HAVE THE FOLLOWING COMMANDS AVAILABLE TO YOU AS CAPTAIN");
        print("OF THE STARSHIP ENTERPRISE:");
        print("");
        # 打印关于NAV命令的说明
        print("\\NAV\\ COMMAND = WARP ENGINE CONTROL --");
        print("     COURSE IS IN A CIRCULAR NUMERICAL      4  3  2");
# 打印向量排列的信息
print("     VECTOR ARRANGEMENT AS SHOWN             . . .");
# 打印可以使用整数和实数值的信息
print("     INTEGER AND REAL VALUES MAY BE           ...");
# 打印数值可以接近9.0的信息
print("     USED.  (THUS COURSE 1.5 IS HALF-     5 ---*--- 1");
# 打印数值可以接近9.0的信息
print("     WAY BETWEEN 1 AND 2                      ...");
# 打印数值可以接近9.0的信息
print("                                             . . .");
# 打印数值可以接近9.0的信息
print("     VALUES MAY APPROACH 9.0, WHICH         6  7  8");
# 打印数值可以接近9.0的信息
print("     ITSELF IS EQUIVALENT TO 1.0");
# 打印星际飞船的航行信息
print("                                            COURSE");
# 打印星际飞船的航行信息
print("     ONE WARP FACTOR IS THE SIZE OF ");
# 打印星际飞船的航行信息
print("     ONE QUADTANT.  THEREFORE, TO GET");
# 打印星际飞船的航行信息
print("     FROM QUADRANT 6,5 TO 5,5, YOU WOULD");
# 打印星际飞船的航行信息
print("     USE COURSE 3, WARP FACTOR 1.");
# 打印空行
print("");
# 打印短程传感器扫描的命令
print("\\SRS\\ COMMAND = SHORT RANGE SENSOR SCAN");
# 打印显示当前象限扫描的信息
print("     SHOWS YOU A SCAN OF YOUR PRESENT QUADRANT.");
# 打印传感器屏幕上的符号含义
print("     SYMBOLOGY ON YOUR SENSOR SCREEN IS AS FOLLOWS:");
# 打印星际飞船的位置符号
print("        <*> = YOUR STARSHIP'S POSITION");
# 打印克林贡战舰的位置符号
print("        +K+ = KLINGON BATTLE CRUISER");
# 打印联邦星舰基地的位置符号
print("        >!< = FEDERATION STARBASE (REFUEL/REPAIR/RE-ARM HERE!)");
        print("         *  = STAR");  # 打印星号表示星星
        print("");  # 打印空行
        print("     A CONDENSED 'STATUS REPORT' WILL ALSO BE PRESENTED.");  # 打印状态报告
        print("");  # 打印空行
        print("\\LRS\\ COMMAND = LONG RANGE SENSOR SCAN");  # 打印LRS命令说明
        print("     SHOWS CONDITIONS IN SPACE FOR ONE QUADRANT ON EACH SIDE");  # 打印LRS命令功能说明
        print("     OF THE ENTERPRISE (WHICH IS IN THE MIDDLE OF THE SCAN)");  # 打印LRS命令功能说明
        print("     THE SCAN IS CODED IN THE FORM \\###\\, WHERE TH UNITS DIGIT");  # 打印LRS命令功能说明
        print("     IS THE NUMBER OF STARS, THE TENS DIGIT IS THE NUMBER OF");  # 打印LRS命令功能说明
        print("     STARBASES, AND THE HUNDRESDS DIGIT IS THE NUMBER OF");  # 打印LRS命令功能说明
        print("     KLINGONS.");  # 打印LRS命令功能说明
        print("");  # 打印空行
        print("     EXAMPLE - 207 = 2 KLINGONS, NO STARBASES, & 7 STARS.");  # 打印LRS命令示例
        print("");  # 打印空行
        print("\\PHA\\ COMMAND = PHASER CONTROL.");  # 打印PHA命令说明
        print("     ALLOWS YOU TO DESTROY THE KLINGON BATTLE CRUISERS BY ");  # 打印PHA命令功能说明
        print("     ZAPPING THEM WITH SUITABLY LARGE UNITS OF ENERGY TO");  # 打印PHA命令功能说明
        print("     DEPLETE THEIR SHIELD POWER.  (REMEMBER, KLINGONS HAVE");  # 打印PHA命令功能说明
        print("     PHASERS TOO!)");  # 打印PHA命令功能说明
        print("");  # 打印空行
# 打印“TOR COMMAND = PHOTON TORPEDO CONTROL”到控制台
print("\\TOR\\ COMMAND = PHOTON TORPEDO CONTROL")
# 打印“TORPEDO COURSE IS THE SAME AS USED IN WARP ENGINE CONTROL”到控制台
print("     TORPEDO COURSE IS THE SAME AS USED IN WARP ENGINE CONTROL")
# 打印“IF YOU HIT THE KLINGON VESSEL, HE IS DESTROYED AND”到控制台
print("     IF YOU HIT THE KLINGON VESSEL, HE IS DESTROYED AND")
# 打印“CANNOT FIRE BACK AT YOU.  IF YOU MISS, YOU ARE SUBJECT TO”到控制台
print("     CANNOT FIRE BACK AT YOU.  IF YOU MISS, YOU ARE SUBJECT TO")
# 打印“HIS PHASER FIRE.  IN EITHER CASE, YOU ARE ALSO SUBJECT TO ”到控制台
print("     HIS PHASER FIRE.  IN EITHER CASE, YOU ARE ALSO SUBJECT TO ")
# 打印“THE PHASER FIRE OF ALL OTHER KLINGONS IN THE QUADRANT.”到控制台
print("     THE PHASER FIRE OF ALL OTHER KLINGONS IN THE QUADRANT.")
# 打印空行到控制台
print("")
# 打印“THE LIBRARY-COMPUTER (\\COM\\ COMMAND) HAS AN OPTION TO ”到控制台
print("     THE LIBRARY-COMPUTER (\\COM\\ COMMAND) HAS AN OPTION TO ")
# 打印“COMPUTE TORPEDO TRAJECTORY FOR YOU (OPTION 2)”到控制台
print("     COMPUTE TORPEDO TRAJECTORY FOR YOU (OPTION 2)")
# 打印空行到控制台
print("")
# 打印“SHE COMMAND = SHIELD CONTROL”到控制台
print("\\SHE\\ COMMAND = SHIELD CONTROL")
# 打印“DEFINES THE NUMBER OF ENERGY UNITS TO BE ASSIGNED TO THE”到控制台
print("     DEFINES THE NUMBER OF ENERGY UNITS TO BE ASSIGNED TO THE")
# 打印“SHIELDS.  ENERGY IS TAKEN FROM TOTAL SHIP'S ENERGY.  NOTE”到控制台
print("     SHIELDS.  ENERGY IS TAKEN FROM TOTAL SHIP'S ENERGY.  NOTE")
# 打印“THAN THE STATUS DISPLAY TOTAL ENERGY INCLUDES SHIELD ENERGY”到控制台
print("     THAN THE STATUS DISPLAY TOTAL ENERGY INCLUDES SHIELD ENERGY")
# 打印空行到控制台
print("")
# 打印“DAM COMMAND = DAMMAGE CONTROL REPORT”到控制台
print("\\DAM\\ COMMAND = DAMMAGE CONTROL REPORT")
# 打印“GIVES THE STATE OF REPAIR OF ALL DEVICES.  WHERE A NEGATIVE”到控制台
print("     GIVES THE STATE OF REPAIR OF ALL DEVICES.  WHERE A NEGATIVE")
# 打印“'STATE OF REPAIR' SHOWS THAT THE DEVICE IS TEMPORARILY”到控制台
print("     'STATE OF REPAIR' SHOWS THAT THE DEVICE IS TEMPORARILY")
# 打印“DAMAGED.”到控制台
print("     DAMAGED.")
# 打印出COM COMMAND = LIBRARY-COMPUTER
print("\\COM\\ COMMAND = LIBRARY-COMPUTER")

# 打印出THE LIBRARY-COMPUTER CONTAINS SIX OPTIONS:
print("     THE LIBRARY-COMPUTER CONTAINS SIX OPTIONS:")

# 打印出OPTION 0 = CUMULATIVE GALACTIC RECORD
print("     OPTION 0 = CUMULATIVE GALACTIC RECORD")

# 打印出THIS OPTION SHOWES COMPUTER MEMORY OF THE RESULTS OF ALL
# PREVIOUS SHORT AND LONG RANGE SENSOR SCANS
print("        THIS OPTION SHOWES COMPUTER MEMORY OF THE RESULTS OF ALL")
print("        PREVIOUS SHORT AND LONG RANGE SENSOR SCANS")

# 打印出OPTION 1 = STATUS REPORT
print("     OPTION 1 = STATUS REPORT")

# 打印出THIS OPTION SHOWS THE NUMBER OF KLINGONS, STARDATES,
# AND STARBASES REMAINING IN THE GAME.
print("        THIS OPTION SHOWS THE NUMBER OF KLINGONS, STARDATES,")
print("        AND STARBASES REMAINING IN THE GAME.")

# 打印出OPTION 2 = PHOTON TORPEDO DATA
print("     OPTION 2 = PHOTON TORPEDO DATA")

# 打印出WHICH GIVES DIRECTIONS AND DISTANCE FROM THE ENTERPRISE
# TO ALL KLINGONS IN YOUR QUADRANT
print("        WHICH GIVES DIRECTIONS AND DISTANCE FROM THE ENTERPRISE")
print("        TO ALL KLINGONS IN YOUR QUADRANT")

# 打印出OPTION 3 = STARBASE NAV DATA
print("     OPTION 3 = STARBASE NAV DATA")

# 打印出THIS OPTION GIVES DIRECTION AND DISTANCE TO ANY 
# STARBASE WITHIN YOUR QUADRANT
print("        THIS OPTION GIVES DIRECTION AND DISTANCE TO ANY ")
print("        STARBASE WITHIN YOUR QUADRANT")

# 打印出OPTION 4 = DIRECTION/DISTANCE CALCULATOR
print("     OPTION 4 = DIRECTION/DISTANCE CALCULATOR")

# 打印出THIS OPTION ALLOWS YOU TO ENTER COORDINATES FOR
# DIRECTION/DISTANCE CALCULATIONS
print("        THIS OPTION ALLOWS YOU TO ENTER COORDINATES FOR")
print("        DIRECTION/DISTANCE CALCULATIONS")

# 打印出OPTION 5 = GALACTIC /REGION NAME/ MAP
print("     OPTION 5 = GALACTIC /REGION NAME/ MAP")

# 打印出THIS OPTION PRINTS THE NAMES OF THE SIXTEEN MAJOR 
# GALACTIC REGIONS REFERRED TO IN THE GAME.
print("        THIS OPTION PRINTS THE NAMES OF THE SIXTEEN MAJOR ")
print("        GALACTIC REGIONS REFERRED TO IN THE GAME.")
    }  // 结束静态方法 print

    static void print(final String s) {  // 定义静态方法 print，用于打印字符串
        System.out.println(s);  // 打印传入的字符串
    }

    static String tab(final int n) {  // 定义静态方法 tab，用于生成指定数量的空格字符串
        return IntStream.range(1, n).mapToObj(num -> " ").collect(Collectors.joining());  // 使用 IntStream 生成指定数量的空格字符串并返回
    }

    static String inputStr(final String message) {  // 定义静态方法 inputStr，用于从控制台输入字符串
        System.out.print(message + "? ");  // 打印提示信息
        final BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));  // 创建 BufferedReader 对象用于读取输入
        try {
            return reader.readLine();  // 读取输入的字符串并返回
        } catch (IOException ioe) {  // 捕获可能的 IO 异常
            ioe.printStackTrace();  // 打印异常信息
            return "";  // 返回空字符串
        }
    }
# 关闭 ZIP 对象
zip.close()
```