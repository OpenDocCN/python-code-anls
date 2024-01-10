# `basic-computer-games\28_Combat\java\Combat.java`

```
    // 导入 java.lang.Math 包，用于数学计算
    import java.lang.Math;
    // 导入 java.util.Scanner 包，用于用户输入
    import java.util.Scanner;

    /**
     * 战斗游戏
     * <p>
     * 基于这里的 BASIC 游戏 Combat
     * https://github.com/coding-horror/basic-computer-games/blob/main/28%20Combat/combat.bas
     * <p>
     * 注意：这个想法是在 Java 中创建一个 1970 年代 BASIC 游戏的版本，不引入新功能 - 没有添加额外的文本、错误检查等。
     *
     * 由 Darren Cardenas 从 BASIC 转换为 Java。
     */
    public class Combat {

      // 每个玩家的最大单位数
      private static final int MAX_UNITS = 72000;

      // 用户输入的扫描器
      private final Scanner scan;

      // 飞机坠毁胜利标志
      private boolean planeCrashWin = false;

      // 用户陆军单位数
      private int usrArmy = 0;
      // 用户海军单位数
      private int usrNavy = 0;
      // 用户空军单位数
      private int usrAir = 0;
      // CPU 陆军单位数
      private int cpuArmy = 30000;
      // CPU 海军单位数
      private int cpuNavy = 20000;
      // CPU 空军单位数
      private int cpuAir = 22000;

      // Combat 类的构造函数
      public Combat() {
        // 初始化用户输入的扫描器
        scan = new Scanner(System.in);
      }  // End of constructor Combat

      // 游戏进行方法
      public void play() {
        // 显示游戏介绍
        showIntro();
        // 获取军队力量
        getForces();
        // 进行第一次攻击
        attackFirst();
        // 进行第二次攻击
        attackSecond();
      }  // End of method play

      // 显示游戏介绍的静态方法
      private static void showIntro() {
        // 打印游戏标题
        System.out.println(" ".repeat(32) + "COMBAT");
        // 打印游戏信息来源
        System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println("\n\n");
        // 打印战争宣言
        System.out.println("I AM AT WAR WITH YOU.");
        // 打印每个玩家的士兵数量
        System.out.println("WE HAVE " + MAX_UNITS + " SOLDIERS APIECE.\n");
      }  // End of method showIntro

      // 获取军队力量的方法
      private void getForces() {
    // 使用 do-while 循环，提示用户分配自己的兵力
    do {
      System.out.println("DISTRIBUTE YOUR FORCES.");
      System.out.println("              ME              YOU");
      System.out.print("ARMY           " + cpuArmy + "        ? ");
      usrArmy = scan.nextInt();
      System.out.print("NAVY           " + cpuNavy + "        ? ");
      usrNavy = scan.nextInt();
      System.out.print("A. F.          " + cpuAir + "        ? ");
      usrAir = scan.nextInt();

    } while ((usrArmy + usrNavy + usrAir) > MAX_UNITS);  // 避免超过最大单位数

  }  // getForces 方法结束

  // attackFirst 方法开始
  private void attackFirst() {

    int numUnits = 0;
    int unitType = 0;

    // 使用 do-while 循环，提示用户选择攻击类型
    do {
      System.out.println("YOU ATTACK FIRST. TYPE (1) FOR ARMY; (2) FOR NAVY;");
      System.out.println("AND (3) FOR AIR FORCE.");
      System.out.print("? ");
      unitType = scan.nextInt();
    } while ((unitType < 1) || (unitType > 3));  // 避免超出范围的值

    // 使用 do-while 循环，提示用户选择攻击单位数量
    do {
      System.out.println("HOW MANY MEN");
      System.out.print("? ");
      numUnits = scan.nextInt();
    } while ((numUnits < 0) ||                // 避免负值
             ((unitType == 1) && (numUnits > usrArmy)) ||  // 避免超过可用的陆军单位数量
             ((unitType == 2) && (numUnits > usrNavy)) ||  // 避免超过可用的海军单位数量
             ((unitType == 3) && (numUnits > usrAir)));    // 避免超过可用的空军单位数量

    // 处理部署类型的开始
    }  // 处理部署类型的结束

  }  // attackFirst 方法结束

  // attackSecond 方法开始
  private void attackSecond() {

    int numUnits = 0;
    int unitType = 0;

    System.out.println("");
    System.out.println("              YOU           ME");
    System.out.print("ARMY           ");
    System.out.format("%-14s%s\n", usrArmy, cpuArmy);
    System.out.print("NAVY           ");
    System.out.format("%-14s%s\n", usrNavy, cpuNavy);
    System.out.print("A. F.          ");
    // 格式化输出用户和CPU的军队数量
    System.out.format("%-14s%s\n", usrAir, cpuAir);

    // 循环直到用户输入合法的单位类型
    do {
      System.out.println("WHAT IS YOUR NEXT MOVE?");
      System.out.println("ARMY=1  NAVY=2  AIR FORCE=3");
      System.out.print("? ");
      unitType = scan.nextInt();
    } while ((unitType < 1) || (unitType > 3));  // 避免超出范围的值

    // 循环直到用户输入合法的单位数量
    do {
      System.out.println("HOW MANY MEN");
      System.out.print("? ");
      numUnits = scan.nextInt();
    } while ((numUnits < 0) ||                // 避免负值
             ((unitType == 1) && (numUnits > usrArmy)) ||  // 避免超出可用陆军单位数量
             ((unitType == 2) && (numUnits > usrNavy)) ||  // 避免超出可用海军单位数量
             ((unitType == 3) && (numUnits > usrAir)));    // 避免超出可用空军单位数量

    // 开始处理部署类型
    # 根据不同的部队类型进行处理
    switch (unitType) {
      case 1:  // Army deployed
        # 如果用户部署的部队数量少于 CPU 部署部队数量的一半
        if (numUnits < (cpuArmy / 2.0)) {  // User deployed less than half relative to cpu Army units
          # 打印信息并更新用户部队数量
          System.out.println("I WIPED OUT YOUR ATTACK!");
          usrArmy = usrArmy - numUnits;
        }
        else {  # 如果用户部署的部队数量等于或多于 CPU 部署部队数量的一半
          # 打印信息并更新 CPU 部队数量
          System.out.println("YOU DESTROYED MY ARMY!");
          cpuArmy = 0;
        }
        break;

      case 2:  // Navy deployed
        # 如果用户部署的部队数量少于 CPU 部署部队数量的一半
        if (numUnits < (cpuNavy / 2.0)) {  // User deployed less than half relative to cpu Navy units
          # 打印信息并更新用户部队数量
          System.out.println("I SUNK TWO OF YOUR BATTLESHIPS, AND MY AIR FORCE");
          System.out.println("WIPED OUT YOUR UNGUARDED CAPITOL.");
          usrArmy = (int) Math.floor(usrArmy / 4.0);
          usrNavy = (int) Math.floor(usrNavy / 2.0);
        }
        else {  # 如果用户部署的部队数量等于或多于 CPU 部署部队数量的一半
          # 打印信息并更新 CPU 部队数量
          System.out.println("YOUR NAVY SHOT DOWN THREE OF MY XIII PLANES,");
          System.out.println("AND SUNK THREE BATTLESHIPS.");
          cpuAir = (int) Math.floor(2.0 * cpuAir / 3.0);
          cpuNavy = (int) Math.floor(cpuNavy / 2.0);
        }
        break;

      case 3:  // Air Force deployed
        # 如果用户部署的部队数量大于 CPU 部署部队数量的一半
        if (numUnits > (cpuAir / 2.0)) {  // User deployed more than half relative to cpu Air Force units
          # 打印信息并更新用户部队数量
          System.out.println("MY NAVY AND AIR FORCE IN A COMBINED ATTACK LEFT");
          System.out.println("YOUR COUNTRY IN SHAMBLES.");
          usrArmy = (int) Math.floor(usrArmy / 3.0);
          usrNavy = (int) Math.floor(usrNavy / 3.0);
          usrAir = (int) Math.floor(usrAir / 3.0);
        }
        else {  # 如果用户部署的部队数量小于或等于 CPU 部署部队数量的一半
          # 打印信息并更新游戏状态
          System.out.println("ONE OF YOUR PLANES CRASHED INTO MY HOUSE. I AM DEAD.");
          System.out.println("MY COUNTRY FELL APART.");
          planeCrashWin = true;
        }
        break;

    }  // End handling deployment type
    // 如果飞机坠毁的消息被抑制
    if (planeCrashWin == false) {
      // 打印空行
      System.out.println("");
      // 打印结果，包括两次攻击的结果
      System.out.println("FROM THE RESULTS OF BOTH OF YOUR ATTACKS,");
    }

    // 用户获胜
    if ((planeCrashWin == true) ||
        ((usrArmy + usrNavy + usrAir) > ((int) Math.floor((3.0 / 2.0 * (cpuArmy + cpuNavy + cpuAir)))))) {
      // 打印用户获胜的消息
      System.out.println("YOU WON, OH! SHUCKS!!!!");
    }
    // 用户失败
    else if ((usrArmy + usrNavy + usrAir) < ((int) Math.floor((2.0 / 3.0 * (cpuArmy + cpuNavy + cpuAir)))) {  // 用户失败
      // 打印用户失败的消息
      System.out.println("YOU LOST-I CONQUERED YOUR COUNTRY.  IT SERVES YOU");
      System.out.println("RIGHT FOR PLAYING THIS STUPID GAME!!!");
    }
    // 和平结果
    else {
      // 打印和平协议的消息
      System.out.println("THE TREATY OF PARIS CONCLUDED THAT WE TAKE OUR");
      System.out.println("RESPECTIVE COUNTRIES AND LIVE IN PEACE.");
    }

  }  // 攻击第二次方法结束

  public static void main(String[] args) {

    // 创建 Combat 对象
    Combat combat = new Combat();
    // 进行游戏
    combat.play();

  }  // 主方法结束
}  # 类 Combat 的结束
```