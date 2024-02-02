# `basic-computer-games\31_Depth_Charge\java\DepthCharge.java`

```py
import java.util.Scanner;  // 导入 Scanner 类，用于用户输入
import java.lang.Math;  // 导入 Math 类，用于数学计算

/**
 * 深水炸弹游戏
 * <p>
 * 基于 BASIC 版本的深水炸弹游戏
 * https://github.com/coding-horror/basic-computer-games/blob/main/31%20Depth%20Charge/depthcharge.bas
 * <p>
 * 注意：本意是在 Java 中创建 1970 年代 BASIC 游戏的版本，没有引入新功能 - 没有添加额外的文本、错误检查等。
 *
 * 由 Darren Cardenas 从 BASIC 转换为 Java。
 */

public class DepthCharge {

  private final Scanner scan;  // 用于用户输入的 Scanner 对象

  public DepthCharge() {

    scan = new Scanner(System.in);  // 初始化 Scanner 对象

  }  // 构造函数 DepthCharge 结束

  public void play() {

    showIntro();  // 显示游戏介绍
    startGame();  // 开始游戏

  }  // 方法 play 结束

  private static void showIntro() {

    System.out.println(" ".repeat(29) + "DEPTH CHARGE");  // 打印游戏标题
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印创意计算公司信息
    System.out.println("\n\n");  // 打印空行

  }  // 方法 showIntro 结束

  private void startGame() {

    int searchArea = 0;  // 搜索区域的维度
    int shotNum = 0;  // 炸弹数量
    int shotTotal = 0;  // 总共的炸弹数量
    int shotX = 0;  // 炸弹的 X 坐标
    int shotY = 0;  // 炸弹的 Y 坐标
    int shotZ = 0;  // 炸弹的 Z 坐标
    int targetX = 0;  // 目标的 X 坐标
    int targetY = 0;  // 目标的 Y 坐标
    int targetZ = 0;  // 目标的 Z 坐标
    int tries = 0;  // 尝试次数
    String[] userCoordinates;  // 用户输入的坐标
    String userResponse = "";  // 用户的响应

    System.out.print("DIMENSION OF SEARCH AREA? ");  // 打印提示信息，要求用户输入搜索区域的维度
    searchArea = Integer.parseInt(scan.nextLine());  // 从用户输入中获取搜索区域的维度
    System.out.println("");  // 打印空行

    shotTotal = (int) (Math.log10(searchArea) / Math.log10(2)) + 1;  // 计算总共的炸弹数量

    System.out.println("YOU ARE THE CAPTAIN OF THE DESTROYER USS COMPUTER");  // 打印游戏背景介绍
    System.out.println("AN ENEMY SUB HAS BEEN CAUSING YOU TROUBLE.  YOUR");
    System.out.println("MISSION IS TO DESTROY IT.  YOU HAVE " + shotTotal + " SHOTS.");
    System.out.println("SPECIFY DEPTH CHARGE EXPLOSION POINT WITH A");
    System.out.println("TRIO OF NUMBERS -- THE FIRST TWO ARE THE");
    System.out.println("SURFACE COORDINATES; THE THIRD IS THE DEPTH.");

    // 开始外部 while 循环
    while (true) {

      // 输出空行
      System.out.println("");
      // 输出"GOOD LUCK !"
      System.out.println("GOOD LUCK !");
      // 输出空行
      System.out.println("");

      // 生成目标坐标X
      targetX = (int) ((searchArea + 1) * Math.random());
      // 生成目标坐标Y
      targetY = (int) ((searchArea + 1) * Math.random());
      // 生成目标坐标Z
      targetZ = (int) ((searchArea + 1) * Math.random());

      // 开始循环射击
      for (shotNum = 1; shotNum <= shotTotal; shotNum++) {

        // 获取用户输入
        System.out.println("");
        System.out.print("TRIAL # " + shotNum + "? ");
        userResponse = scan.nextLine();

        // 以逗号分割用户输入
        userCoordinates = userResponse.split(",");

        // 赋值给整数变量
        shotX = Integer.parseInt(userCoordinates[0].trim());
        shotY = Integer.parseInt(userCoordinates[1].trim());
        shotZ = Integer.parseInt(userCoordinates[2].trim());

        // 胜利条件
        if (Math.abs(shotX - targetX) + Math.abs(shotY - targetY)
            + Math.abs(shotZ - targetZ) == 0) {

          // 输出"BOOM! YOU FOUND IT IN [shotNum] TRIES!"
          System.out.println("B O O M ! ! YOU FOUND IT IN" + shotNum + " TRIES!");
          // 跳出循环
          break;

        }

        // 调用getReport方法
        this.getReport(targetX, targetY, targetZ, shotX, shotY, shotZ);

        // 输出空行
        System.out.println("");

      }  // 结束循环射击

      // 如果射击次数超过总次数
      if (shotNum > shotTotal) {

        // 输出空行
        System.out.println("");
        // 输出"YOU HAVE BEEN TORPEDOED!  ABANDON SHIP!"
        System.out.println("YOU HAVE BEEN TORPEDOED!  ABANDON SHIP!");
        // 输出"THE SUBMARINE WAS AT [targetX],[targetY],[targetZ]"
        System.out.println("THE SUBMARINE WAS AT " + targetX + "," + targetY + "," + targetZ);
      }

      // 输出空行
      System.out.println("");
      // 输出空行
      System.out.println("");
      // 输出"ANOTHER GAME (Y OR N)? "
      System.out.print("ANOTHER GAME (Y OR N)? ");
      userResponse = scan.nextLine();

      // 如果用户输入不是"Y"，则结束游戏
      if (!userResponse.toUpperCase().equals("Y")) {
        // 输出"OK.  HOPE YOU ENJOYED YOURSELF."
        System.out.print("OK.  HOPE YOU ENJOYED YOURSELF.");
        // 返回
        return;
      }

    }  // 结束外部while循环

  }  // 结束startGame方法

  // 定义getReport方法，用于获取报告
  public void getReport(int a, int b, int c, int x, int y, int z) {

    // 输出"SONAR REPORTS SHOT WAS "
    System.out.print("SONAR REPORTS SHOT WAS ");

    // 处理y坐标
    // 如果y坐标大于b坐标，打印"NORTH"
    if (y > b) {

      System.out.print("NORTH");

    } else if (y < b) {  // 如果y坐标小于b坐标，打印"SOUTH"

      System.out.print("SOUTH");
    }

    // 处理x坐标
    if (x > a) {  // 如果x坐标大于a坐标，打印"EAST"

      System.out.print("EAST");

    } else if (x < a) {  // 如果x坐标小于a坐标，打印"WEST"

      System.out.print("WEST");
    }

    // 如果y坐标不等于b坐标或者x坐标不等于a坐标，打印" AND"
    if ((y != b) || (x != a)) {

      System.out.print(" AND");
    }

    // 处理深度
    if (z > c) {  // 如果z坐标大于c坐标，打印" TOO LOW."

      System.out.println(" TOO LOW.");

    } else  if (z < c) {  // 如果z坐标小于c坐标，打印" TOO HIGH."

      System.out.println(" TOO HIGH.");

    } else {  // 否则，打印" DEPTH OK."

      System.out.println(" DEPTH OK.");
    }

    return;

  }  // 方法getReport结束

  public static void main(String[] args) {

    DepthCharge game = new DepthCharge();
    game.play();

  }  // 方法main结束
}  // 类 DepthCharge 结束
```