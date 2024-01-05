# `13_Bounce\java\Bounce.java`

```
import java.util.Scanner;  # 导入 Scanner 类，用于用户输入
import java.lang.Math;  # 导入 Math 类，用于数学运算

/**
 * Game of Bounce
 * <p>
 * Based on the BASIC game of Bounce here
 * https://github.com/coding-horror/basic-computer-games/blob/main/13%20Bounce/bounce.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

public class Bounce {

  private final Scanner scan;  // 用于用户输入的 Scanner 对象

  public Bounce() {
    scan = new Scanner(System.in);  // 创建一个新的 Scanner 对象，用于从控制台读取输入

  }  // End of constructor Bounce

  public void play() {

    showIntro();  // 调用 showIntro 方法显示游戏介绍
    startGame();  // 调用 startGame 方法开始游戏

  }  // End of method play

  private void showIntro() {

    System.out.println(" ".repeat(32) + "BOUNCE");  // 在控制台打印游戏标题
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 在控制台打印游戏信息
    System.out.println("\n\n");  // 在控制台打印空行

  }  // End of method showIntro
    // 初始化变量
    double coefficient = 0; // 初始化弹性系数
    double height = 0; // 初始化高度
    double timeIncrement = 0; // 初始化时间增量
    double timeIndex = 0; // 初始化时间索引
    double timeTotal = 0; // 初始化总时间
    double velocity = 0; // 初始化速度

    double[] timeData = new double[21]; // 创建长度为21的时间数据数组

    int heightInt = 0; // 初始化整数高度
    int index = 0; // 初始化索引
    int maxData = 0; // 初始化最大数据

    String lineContent = ""; // 初始化行内容

    // 输出提示信息
    System.out.println("THIS SIMULATION LETS YOU SPECIFY THE INITIAL VELOCITY");
    System.out.println("OF A BALL THROWN STRAIGHT UP, AND THE COEFFICIENT OF");
    System.out.println("ELASTICITY OF THE BALL.  PLEASE USE A DECIMAL FRACTION");
    # 打印"COEFFICIENCY (LESS THAN 1)."到控制台
    print("COEFFICIENCY (LESS THAN 1).")
    # 打印空行到控制台
    print("")
    # 打印"YOU ALSO SPECIFY THE TIME INCREMENT TO BE USED IN"到控制台
    print("YOU ALSO SPECIFY THE TIME INCREMENT TO BE USED IN")
    # 打印"'STROBING' THE BALL'S FLIGHT (TRY .1 INITIALLY)."到控制台
    print("'STROBING' THE BALL'S FLIGHT (TRY .1 INITIALLY).")
    # 打印空行到控制台
    print("")

    # 开始外部 while 循环
    while True:

      # 打印"TIME INCREMENT (SEC)? "到控制台，并等待用户输入
      timeIncrement = float(input("TIME INCREMENT (SEC)? "))
      # 打印空行到控制台
      print("")

      # 打印"VELOCITY (FPS)? "到控制台，并等待用户输入
      velocity = float(input("VELOCITY (FPS)? "))
      # 打印空行到控制台
      print("")

      # 打印"COEFFICIENT? "到控制台，并等待用户输入
      coefficient = float(input("COEFFICIENT? "))
      # 打印空行到控制台
      print("")
      // 打印"FEET"
      System.out.println("FEET");
      // 打印空行
      System.out.println("");

      // 计算最大数据点数
      maxData = (int)(70 / (velocity / (16 * timeIncrement)));

      // 循环计算时间数据
      for (index = 1; index <= maxData; index++) {
        timeData[index] = velocity * Math.pow(coefficient, index - 1) / 16;
      }

      // 开始循环遍历y轴数据的所有行
      for (heightInt = (int)(-16 * Math.pow(velocity / 32, 2) + Math.pow(velocity, 2) / 32 + 0.5) * 10;
           heightInt >= 0; heightInt -= 5) {

        height = heightInt / 10.0;

        lineContent = "";

        // 如果高度是整数
        if ((int)(Math.floor(height)) == height) {
          lineContent += " " + (int)(height) + " ";  // 将当前高度转换为整数并添加到lineContent中
        }

        timeTotal = 0;  // 初始化时间总和为0

        for (index = 1; index <= maxData; index++) {  // 遍历数据点

          for (timeIndex = 0; timeIndex <= timeData[index]; timeIndex += timeIncrement) {  // 遍历时间数据

            timeTotal += timeIncrement;  // 增加时间总和

            if (Math.abs(height - (0.5 * (-32) * Math.pow(timeIndex, 2) + velocity
                * Math.pow(coefficient, index - 1) * timeIndex)) <= 0.25) {  // 判断当前高度是否在合适范围内

              while (lineContent.length() < (timeTotal / timeIncrement) - 1) {  // 当lineContent长度小于当前时间总和除以时间增量减1时
                lineContent += " ";  // 添加空格
              }
              lineContent += "0";  // 添加0
            }
          }
      timeIndex = timeData[index + 1] / 2;  # 计算时间索引，将时间数据数组中的下一个值除以2

      if (-16 * Math.pow(timeIndex, 2) + velocity * Math.pow(coefficient, index - 1) * timeIndex < height) {  # 如果给定条件成立
        break;  # 跳出循环
      }
    }

    System.out.println(lineContent);  # 打印行内容

  }  // End loop through all rows of y-axis data  # 结束循环，遍历所有y轴数据的行

  lineContent = "";  # 重置行内容为空

  // Show the x-axis  # 显示x轴
  for (index = 1; index <= (int)(timeTotal + 1) / timeIncrement + 1; index++) {  # 循环遍历x轴数据
    lineContent += ".";  # 将点添加到行内容中
  }
      System.out.println(lineContent);  // 打印变量 lineContent 的内容

      lineContent = " 0";  // 将变量 lineContent 的值设为 " 0"

      for (index = 1; index <= (int)(timeTotal + 0.9995); index++) {  // 循环，index 从 1 到 timeTotal 的整数部分加上 0.9995
        while (lineContent.length() < (int)(index / timeIncrement)) {  // 当 lineContent 的长度小于 index 除以 timeIncrement 的整数部分时执行循环
          lineContent += " ";  // 在 lineContent 后面添加一个空格
        }
        lineContent += index;  // 将 index 添加到 lineContent 的末尾
      }

      System.out.println(lineContent);  // 打印变量 lineContent 的内容

      System.out.println(" ".repeat((int)((timeTotal + 1) / (2 * timeIncrement) - 3)) + "SECONDS");  // 打印一定数量的空格再加上 "SECONDS"

    }  // 结束外部 while 循环

  }  // 方法 startGame 结束
# 定义一个名为main的公共静态方法，参数为字符串数组args
def main(args):
    # 创建一个名为game的Bounce对象
    game = Bounce()
    # 调用Bounce对象的play方法
    game.play()
  # main方法结束

# Bounce类结束
```