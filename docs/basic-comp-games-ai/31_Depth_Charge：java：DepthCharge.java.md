# `31_Depth_Charge\java\DepthCharge.java`

```
import java.util.Scanner;  # 导入 java.util.Scanner 包，用于用户输入
import java.lang.Math;  # 导入 java.lang.Math 包，用于数学运算

/**
 * 深水炸弹游戏
 * <p>
 * 基于 BASIC 版本的深水炸弹游戏，链接在这里
 * https://github.com/coding-horror/basic-computer-games/blob/main/31%20Depth%20Charge/depthcharge.bas
 * <p>
 * 注意：这个想法是在 Java 中创建一个 1970 年代 BASIC 游戏的版本，没有引入新功能 - 没有添加额外的文本、错误检查等。
 *
 * 由 Darren Cardenas 从 BASIC 转换为 Java。
 */

public class DepthCharge {

  private final Scanner scan;  // 用于用户输入的扫描器

  public DepthCharge() {
    scan = new Scanner(System.in);  // 创建一个新的 Scanner 对象，用于从标准输入读取用户输入

  }  // End of constructor DepthCharge  // DepthCharge 构造函数的结束

  public void play() {  // play 方法的开始

    showIntro();  // 调用 showIntro 方法显示游戏介绍
    startGame();  // 调用 startGame 方法开始游戏

  }  // End of method play  // play 方法的结束

  private static void showIntro() {  // showIntro 方法的开始

    System.out.println(" ".repeat(29) + "DEPTH CHARGE");  // 打印游戏标题
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印游戏信息
    System.out.println("\n\n");  // 打印空行

  }  // End of method showIntro  // showIntro 方法的结束
    # 初始化搜索区域大小、射击次数、总射击次数、射击坐标、目标坐标、尝试次数、用户输入的坐标数组和用户响应字符串
    searchArea = 0
    shotNum = 0
    shotTotal = 0
    shotX = 0
    shotY = 0
    shotZ = 0
    targetX = 0
    targetY = 0
    targetZ = 0
    tries = 0
    userCoordinates = []
    userResponse = ""

    # 提示用户输入搜索区域的维度，并将用户输入的字符串转换为整数赋值给searchArea
    System.out.print("DIMENSION OF SEARCH AREA? ")
    searchArea = Integer.parseInt(scan.nextLine())
    System.out.println("")

    # 根据搜索区域的大小计算总射击次数，使用对数函数计算
    shotTotal = (int) (Math.log10(searchArea) / Math.log10(2)) + 1;
    # 打印出指挥官的信息
    print("YOU ARE THE CAPTAIN OF THE DESTROYER USS COMPUTER")
    print("AN ENEMY SUB HAS BEEN CAUSING YOU TROUBLE.  YOUR")
    print("MISSION IS TO DESTROY IT.  YOU HAVE " + shotTotal + " SHOTS.")
    print("SPECIFY DEPTH CHARGE EXPLOSION POINT WITH A")
    print("TRIO OF NUMBERS -- THE FIRST TWO ARE THE")
    print("SURFACE COORDINATES; THE THIRD IS THE DEPTH.")

    # 开始外部循环
    while True:

      print("")
      print("GOOD LUCK !")
      print("")

      targetX = (int) ((searchArea + 1) * Math.random())
      targetY = (int) ((searchArea + 1) * Math.random())
      targetZ = (int) ((searchArea + 1) * Math.random())

      # 开始循环遍历所有射击
      for (shotNum = 1; shotNum <= shotTotal; shotNum++) {
        // 循环，从第一次射击到总射击次数

        // 获取用户输入
        System.out.println("");
        System.out.print("TRIAL # " + shotNum + "? ");
        userResponse = scan.nextLine();

        // 以逗号分隔用户输入
        userCoordinates = userResponse.split(",");

        // 赋值给整数变量
        shotX = Integer.parseInt(userCoordinates[0].trim());
        shotY = Integer.parseInt(userCoordinates[1].trim());
        shotZ = Integer.parseInt(userCoordinates[2].trim());

        // 胜利条件
        if (Math.abs(shotX - targetX) + Math.abs(shotY - targetY)
            + Math.abs(shotZ - targetZ) == 0) {
          // 如果射击坐标与目标坐标完全匹配
          System.out.println("B O O M ! ! YOU FOUND IT IN" + shotNum + " TRIES!");
          break;  // 结束循环

        }

        this.getReport(targetX, targetY, targetZ, shotX, shotY, shotZ);  // 调用getReport方法，传入目标和射击坐标

        System.out.println("");  // 输出空行

      }  // 循环遍历所有射击结束

      if (shotNum > shotTotal) {  // 如果射击次数大于总次数

        System.out.println("");  // 输出空行
        System.out.println("YOU HAVE BEEN TORPEDOED!  ABANDON SHIP!");  // 输出提示信息
        System.out.println("THE SUBMARINE WAS AT " + targetX + "," + targetY + "," + targetZ);  // 输出潜艇的位置信息
      }

      System.out.println("");  // 输出空行
      System.out.println("");  // 输出空行
      System.out.print("ANOTHER GAME (Y OR N)? ");  // 提示用户是否开始另一局游戏
      userResponse = scan.nextLine();  // 从用户输入中读取一行字符串

      if (!userResponse.toUpperCase().equals("Y")) {  // 如果用户输入的字符串转换为大写后不等于"Y"
        System.out.print("OK.  HOPE YOU ENJOYED YOURSELF.");  // 输出提示信息
        return;  // 结束方法
      }

    }  // 结束外部的 while 循环

  }  // 结束 startGame 方法

  public void getReport(int a, int b, int c, int x, int y, int z) {  // 定义一个名为 getReport 的方法，接受六个整数参数

    System.out.print("SONAR REPORTS SHOT WAS ");  // 输出提示信息

    // 处理 y 坐标
    if (y > b) {  // 如果 y 坐标大于 b

      System.out.print("NORTH");  // 输出提示信息 "NORTH"
    } else if (y < b) {  // 如果当前坐标的 y 值小于目标坐标的 y 值
      System.out.print("SOUTH");  // 输出"SOUTH"
    }

    // 处理 x 坐标
    if (x > a) {  // 如果当前坐标的 x 值大于目标坐标的 x 值
      System.out.print("EAST");  // 输出"EAST"
    } else if (x < a) {  // 如果当前坐标的 x 值小于目标坐标的 x 值
      System.out.print("WEST");  // 输出"WEST"
    }

    if ((y != b) || (x != a)) {  // 如果当前坐标的 y 值不等于目标坐标的 y 值 或者 当前坐标的 x 值不等于目标坐标的 x 值
      System.out.print(" AND");  // 输出" AND"
    }
    // 处理深度
    if (z > c) {
      // 如果猜测值大于目标值，打印提示信息
      System.out.println(" TOO LOW.");
    } else  if (z < c) {
      // 如果猜测值小于目标值，打印提示信息
      System.out.println(" TOO HIGH.");
    } else {
      // 如果猜测值等于目标值，打印提示信息
      System.out.println(" DEPTH OK.");
    }
    // 返回
    return;
  }  // 方法 getReport 结束

  public static void main(String[] args) {
    DepthCharge game = new DepthCharge();  // 创建一个名为game的DepthCharge对象
    game.play();  // 调用DepthCharge对象的play方法开始游戏

  }  // End of method main  // main方法结束

}  // End of class DepthCharge  // DepthCharge类结束
```