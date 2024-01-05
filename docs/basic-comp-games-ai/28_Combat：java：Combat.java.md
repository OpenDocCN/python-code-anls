# `d:/src/tocomm/basic-computer-games\28_Combat\java\Combat.java`

```
import java.lang.Math;  // 导入 java.lang.Math 类，用于数学运算
import java.util.Scanner;  // 导入 java.util.Scanner 类，用于接收用户输入

/**
 * Game of Combat
 * <p>
 * Based on the BASIC game of Combat here
 * https://github.com/coding-horror/basic-computer-games/blob/main/28%20Combat/combat.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

public class Combat {

  private static final int MAX_UNITS = 72000;  // 定义最大单位数为 72000，每个玩家的最大总单位数

  private final Scanner scan;  // 创建 Scanner 对象，用于接收用户输入
  private boolean planeCrashWin = false;  // 初始化一个布尔变量，表示飞机坠毁是否获胜

  private int usrArmy = 0;      // 用户陆军单位的数量
  private int usrNavy = 0;      // 用户海军单位的数量
  private int usrAir = 0;       // 用户空军单位的数量
  private int cpuArmy = 30000;  // CPU陆军单位的数量
  private int cpuNavy = 20000;  // CPU海军单位的数量
  private int cpuAir = 22000;   // CPU空军单位的数量

  public Combat() {
    scan = new Scanner(System.in);  // 创建一个用于接收用户输入的Scanner对象
  }  // 构造函数 Combat 的结束

  public void play() {
    showIntro();  // 调用显示游戏介绍的方法
    getForces();  // 调用获取军队数量的方法
  }
    attackFirst();
    attackSecond();
    // 调用攻击方法，先攻击对手的第一个单位
    // 调用攻击方法，再攻击对手的第二个单位

  }  // End of method play
  // play 方法结束

  private static void showIntro() {
    // 显示游戏介绍
    System.out.println(" ".repeat(32) + "COMBAT");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");
    System.out.println("I AM AT WAR WITH YOU.");
    System.out.println("WE HAVE " + MAX_UNITS + " SOLDIERS APIECE.\n");

  }  // End of method showIntro
  // showIntro 方法结束

  private void getForces() {
    // 获取双方的军队力量
    do {
      System.out.println("DISTRIBUTE YOUR FORCES.");
      System.out.println("              ME              YOU");
      // 提示玩家分配军队力量
      System.out.print("ARMY           " + cpuArmy + "        ? ");  // 输出提示信息，显示对方陆军数量，并等待用户输入
      usrArmy = scan.nextInt();  // 从用户输入获取对方陆军数量
      System.out.print("NAVY           " + cpuNavy + "        ? ");  // 输出提示信息，显示对方海军数量，并等待用户输入
      usrNavy = scan.nextInt();  // 从用户输入获取对方海军数量
      System.out.print("A. F.          " + cpuAir + "        ? ");  // 输出提示信息，显示对方空军数量，并等待用户输入
      usrAir = scan.nextInt();  // 从用户输入获取对方空军数量

    } while ((usrArmy + usrNavy + usrAir) > MAX_UNITS);  // 当用户输入的总兵力数量超过最大允许数量时，继续循环，直到输入符合要求为止

  }  // 方法 getForces 结束

  private void attackFirst() {

    int numUnits = 0;  // 初始化变量 numUnits 为 0
    int unitType = 0;  // 初始化变量 unitType 为 0

    do {
      System.out.println("YOU ATTACK FIRST. TYPE (1) FOR ARMY; (2) FOR NAVY;");  // 输出提示信息，提示用户输入 1 选择陆军，2 选择海军
      System.out.println("AND (3) FOR AIR FORCE.");  // 输出提示信息，提示用户输入 3 选择空军
      System.out.print("? ");  // 输出提示信息，等待用户输入
      unitType = scan.nextInt(); // 从用户输入中读取单位类型

    } while ((unitType < 1) || (unitType > 3));  // 避免超出范围的值

    do {
      System.out.println("HOW MANY MEN"); // 打印提示信息
      System.out.print("? "); // 打印提示信息
      numUnits = scan.nextInt(); // 从用户输入中读取单位数量
    } while ((numUnits < 0) ||                // 避免负值
             ((unitType == 1) && (numUnits > usrArmy)) ||  // 避免超出可用陆军单位数量
             ((unitType == 2) && (numUnits > usrNavy)) ||  // 避免超出可用海军单位数量
             ((unitType == 3) && (numUnits > usrAir)));    // 避免超出可用空军单位数量

    // 开始处理部署类型
    switch (unitType) {
      case 1:  // 部署陆军

        if (numUnits < (usrArmy / 3.0)) {  // 用户部署的陆军单位少于总陆军单位的三分之一
          System.out.println("YOU LOST " + numUnits + " MEN FROM YOUR ARMY."); // 打印提示信息
          usrArmy = usrArmy - numUnits; // 更新剩余陆军单位数量
        }
        else if (numUnits < (2.0 * usrArmy / 3.0)) {  // 如果用户部署的单位少于其军队单位的三分之二
          System.out.println("YOU LOST " + (int) Math.floor(numUnits / 3.0) + " MEN, BUT I LOST " + (int) Math.floor(2.0 * cpuArmy / 3.0));
          usrArmy = (int) Math.floor(usrArmy - numUnits / 3.0);
          cpuArmy = 0;
        }
        else {  // 如果用户部署的单位达到或超过其军队单位的三分之二
          System.out.println("YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO");
          System.out.println("OF YOUR AIR FORCE BASES AND 3 ARMY BASES.");
          usrArmy = (int) Math.floor(usrArmy / 3.0);
          usrAir = (int) Math.floor(usrAir / 3.0);
          cpuNavy = (int) Math.floor(2.0 * cpuNavy / 3.0);
        }
        break;

      case 2:  // 海军部署

        if (numUnits < (cpuNavy / 3.0)) {  // 如果用户部署的单位少于相对于 CPU 海军单位的三分之一
          System.out.println("YOUR ATTACK WAS STOPPED!");
          usrNavy = usrNavy - numUnits;
        }
        else if (numUnits < (2.0 * cpuNavy / 3.0)) {  // 如果用户部署的单位数量小于 CPU 海军单位的三分之二
          System.out.println("YOU DESTROYED " + (int) Math.floor(2.0 * cpuNavy / 3.0) + " OF MY ARMY.");  // 打印消息，表示用户摧毁了 CPU 陆军的部分
          cpuNavy = (int) Math.floor(cpuNavy / 3.0);  // 更新 CPU 海军单位数量
        }
        else {  // 如果用户部署的单位数量达到或超过 CPU 海军单位的三分之二
          System.out.println("YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO");  // 打印消息，表示用户击沉了 CPU 的巡逻艇，但 CPU 摧毁了用户的两个空军基地和三个陆军基地
          System.out.println("OF YOUR AIR FORCE BASES AND 3 ARMY BASES.");
          usrArmy = (int) Math.floor(usrArmy / 3.0);  // 更新用户陆军单位数量
          usrAir = (int) Math.floor(usrAir / 3.0);  // 更新用户空军单位数量
          cpuNavy = (int) Math.floor(2.0 * cpuNavy / 3.0);  // 更新 CPU 海军单位数量
        }
        break;

      case 3:  // 空军部署

        if (numUnits < (usrAir / 3.0)) {  // 如果用户部署的单位数量小于用户空军单位的三分之一
          System.out.println("YOUR ATTACK WAS WIPED OUT.");  // 打印消息，表示用户的攻击被摧毁
          usrAir = usrAir - numUnits;  // 更新用户空军单位数量
        }
        else if (numUnits < (2.0 * usrAir / 3.0)) {  // 如果用户部署的单位数量小于用户空军单位的三分之二
System.out.println("WE HAD A DOGFIGHT. YOU WON - AND FINISHED YOUR MISSION.");
// 打印消息，表示用户在空战中获胜并完成了任务
cpuArmy = (int) Math.floor(2.0 * cpuArmy / 3.0);
// 计算并更新敌方陆军数量，将其减少到原来的三分之二
cpuNavy = (int) Math.floor(cpuNavy / 3.0);
// 计算并更新敌方海军数量，将其减少到原来的三分之一
cpuAir = (int) Math.floor(cpuAir / 3.0);
// 计算并更新敌方空军数量，将其减少到原来的三分之一
}
else {  // User deployed two-thirds or more of their Air Force units
// 如果用户部署了三分之二或更多的空军单位
System.out.println("YOU WIPED OUT ONE OF MY ARMY PATROLS, BUT I DESTROYED");
// 打印消息，表示用户摧毁了敌方陆军巡逻队之一
System.out.println("TWO NAVY BASES AND BOMBED THREE ARMY BASES.");
// 打印消息，表示用户摧毁了两个海军基地并轰炸了三个陆军基地
usrArmy = (int) Math.floor(usrArmy / 4.0);
// 计算并更新用户陆军数量，将其减少到原来的四分之一
usrNavy = (int) Math.floor(usrNavy / 3.0);
// 计算并更新用户海军数量，将其减少到原来的三分之一
cpuArmy = (int) Math.floor(2.0 * cpuArmy / 3.0);
// 计算并更新敌方陆军数量，将其减少到原来的三分之二
}
break;
// 结束处理部署类型
}  // End of method attackFirst
// 结束 attackFirst 方法

private void attackSecond() {
// 定义 attackSecond 方法
    int numUnits = 0;  // 初始化一个整型变量 numUnits，用于存储单位数量
    int unitType = 0;  // 初始化一个整型变量 unitType，用于存储单位类型

    System.out.println("");  // 打印空行
    System.out.println("              YOU           ME");  // 打印表头
    System.out.print("ARMY           ");  // 打印单位类型
    System.out.format("%-14s%s\n", usrArmy, cpuArmy);  // 打印用户和CPU的陆军数量
    System.out.print("NAVY           ");  // 打印单位类型
    System.out.format("%-14s%s\n", usrNavy, cpuNavy);  // 打印用户和CPU的海军数量
    System.out.print("A. F.          ");  // 打印单位类型
    System.out.format("%-14s%s\n", usrAir, cpuAir);  // 打印用户和CPU的空军数量

    do {
      System.out.println("WHAT IS YOUR NEXT MOVE?");  // 打印提示信息
      System.out.println("ARMY=1  NAVY=2  AIR FORCE=3");  // 打印单位类型选择提示
      System.out.print("? ");  // 打印提示符
      unitType = scan.nextInt();  // 从用户输入中读取单位类型
    } while ((unitType < 1) || (unitType > 3));  // 当单位类型不在范围内时，继续循环，避免超出范围的值

    do {
      System.out.println("HOW MANY MEN");  // 输出提示信息，询问用户要部署多少单位
      System.out.print("? ");  // 输出提示信息，等待用户输入
      numUnits = scan.nextInt();  // 从用户输入中获取要部署的单位数量
    } while ((numUnits < 0) ||                // 避免输入负值
             ((unitType == 1) && (numUnits > usrArmy)) ||  // 避免超过可用陆军单位数量
             ((unitType == 2) && (numUnits > usrNavy)) ||  // 避免超过可用海军单位数量
             ((unitType == 3) && (numUnits > usrAir)));    // 避免超过可用空军单位数量

    // 开始处理部署类型
    switch (unitType) {
      case 1:  // 部署陆军

        if (numUnits < (cpuArmy / 2.0)) {  // 用户部署的数量少于对手陆军数量的一半
          System.out.println("I WIPED OUT YOUR ATTACK!");  // 输出提示信息，对手摧毁了用户的进攻
          usrArmy = usrArmy - numUnits;  // 减去用户部署的陆军数量
        }
        else {  // 用户部署的数量达到或超过对手陆军数量的一半
          System.out.println("YOU DESTROYED MY ARMY!");  // 输出提示信息，用户摧毁了对手的陆军
          cpuArmy = 0;  // 对手陆军数量清零
        }
        break;  // 结束当前的 case 分支

      case 2:  // Navy deployed  // 情况2：海军部署

        if (numUnits < (cpuNavy / 2.0)) {  // 如果用户部署的单位数量少于 CPU 海军单位的一半
          System.out.println("I SUNK TWO OF YOUR BATTLESHIPS, AND MY AIR FORCE");  // 打印消息
          System.out.println("WIPED OUT YOUR UNGUARDED CAPITOL.");  // 打印消息
          usrArmy = (int) Math.floor(usrArmy / 4.0);  // 用户陆军数量减少为当前数量的四分之一
          usrNavy = (int) Math.floor(usrNavy / 2.0);  // 用户海军数量减少为当前数量的一半
        }
        else { // 如果用户部署的单位数量等于或多于 CPU 海军单位的一半
          System.out.println("YOUR NAVY SHOT DOWN THREE OF MY XIII PLANES,");  // 打印消息
          System.out.println("AND SUNK THREE BATTLESHIPS.");  // 打印消息
          cpuAir = (int) Math.floor(2.0 * cpuAir / 3.0);  // CPU 空军数量减少为当前数量的三分之二
          cpuNavy = (int) Math.floor(cpuNavy / 2.0);  // CPU 海军数量减少为当前数量的一半
        }
        break;  // 结束当前的 case 分支

      case 3:  // Air Force deployed  // 情况3：空军部署
        if (numUnits > (cpuAir / 2.0)) {  // 如果用户部署的单位数量超过了 CPU 空军单位的一半
          System.out.println("MY NAVY AND AIR FORCE IN A COMBINED ATTACK LEFT");  // 打印消息：我的海军和空军联合进攻
          System.out.println("YOUR COUNTRY IN SHAMBLES.");  // 打印消息：你的国家支离破碎
          usrArmy = (int) Math.floor(usrArmy / 3.0);  // 用户陆军数量减少为原来的三分之一
          usrNavy = (int) Math.floor(usrNavy / 3.0);  // 用户海军数量减少为原来的三分之一
          usrAir = (int) Math.floor(usrAir / 3.0);  // 用户空军数量减少为原来的三分之一
        }
        else {  // 如果用户部署的单位数量等于或少于 CPU 空军单位的一半
          System.out.println("ONE OF YOUR PLANES CRASHED INTO MY HOUSE. I AM DEAD.");  // 打印消息：你的一架飞机撞到了我的房子。我死了。
          System.out.println("MY COUNTRY FELL APART.");  // 打印消息：我的国家支离破碎
          planeCrashWin = true;  // 设置飞机坠毁胜利标志为真
        }
        break;  // 结束处理部署类型

    }  // 结束 if-else 语句块

    // 抑制飞机坠毁的消息
    if (planeCrashWin == false) {  // 如果飞机坠毁胜利标志为假
      System.out.println("");  // 打印空行
      System.out.println("FROM THE RESULTS OF BOTH OF YOUR ATTACKS,");  // 打印消息：从你们两次进攻的结果来看，
    // User wins
    // 如果飞机坠毁胜利为真，或者用户陆军、海军、空军总数大于对手总数的3/2向下取整
    if ((planeCrashWin == true) ||
        ((usrArmy + usrNavy + usrAir) > ((int) Math.floor((3.0 / 2.0 * (cpuArmy + cpuNavy + cpuAir))))) {
      // 打印用户获胜信息
      System.out.println("YOU WON, OH! SHUCKS!!!!");
    }
    // User loses
    // 如果用户陆军、海军、空军总数小于对手总数的2/3向下取整
    else if ((usrArmy + usrNavy + usrAir) < ((int) Math.floor((2.0 / 3.0 * (cpuArmy + cpuNavy + cpuAir)))) {  // User loss
      // 打印用户失败信息
      System.out.println("YOU LOST-I CONQUERED YOUR COUNTRY.  IT SERVES YOU");
      System.out.println("RIGHT FOR PLAYING THIS STUPID GAME!!!");
    }
    // Peaceful outcome
    // 其他情况下
    else {
      // 打印和平结果信息
      System.out.println("THE TREATY OF PARIS CONCLUDED THAT WE TAKE OUR");
      System.out.println("RESPECTIVE COUNTRIES AND LIVE IN PEACE.");
    }

  }  // End of method attackSecond
# 定义一个名为 main 的公共静态方法，参数为字符串数组 args
public static void main(String[] args) {
    # 创建一个名为 combat 的 Combat 对象
    Combat combat = new Combat();
    # 调用 Combat 对象的 play 方法
    combat.play();
}  # 方法 main 结束

}  # 类 Combat 结束
```