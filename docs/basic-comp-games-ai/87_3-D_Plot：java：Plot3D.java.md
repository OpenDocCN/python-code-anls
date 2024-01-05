# `d:/src/tocomm/basic-computer-games\87_3-D_Plot\java\Plot3D.java`

```
import java.lang.Math;  # 导入 java.lang.Math 包，用于数学计算

/**
 * Game of 3-D Plot
 * <p>
 * Based on the BASIC game of 3-D Plot here
 * https://github.com/coding-horror/basic-computer-games/blob/main/87%203-D%20Plot/3dplot.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

// Java class names cannot begin with a letter, so class name 3dplot cannot be used
public class Plot3D {  # 定义名为 Plot3D 的公共类

  public void play() {  # 定义名为 play 的公共方法
    showIntro();  # 调用showIntro()方法，显示游戏介绍

    startGame();  # 调用startGame()方法，开始游戏

  }  // End of method play  # play方法结束


  private void showIntro() {  # 定义showIntro()方法

    System.out.println(" ".repeat(31) + "3D PLOT");  # 在控制台打印"3D PLOT"，并在前面添加31个空格

    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  # 在控制台打印"CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并在前面添加14个空格

    System.out.println("\n\n\n");  # 在控制台打印三个换行符

  }  // End of method showIntro  # showIntro方法结束


  private void startGame() {  # 定义startGame()方法

    float row = 0;  # 初始化浮点型变量row为0
    int column = 0;  # 初始化整型变量column为0
    int limit = 0;  # 初始化整型变量limit为0
    int plotVal = 0;  // 初始化变量plotVal，用于存储计算得到的绘图数值
    int root = 0;  // 初始化变量root，用于存储计算得到的平方根值

    String lineContent = "";  // 初始化变量lineContent，用于存储字符串内容

    // 开始循环遍历所有行
    for (row = -30; row <= 30; row += 1.5) {

      limit = 0;  // 初始化变量limit，用于存储限制值

      root = 5 * (int) Math.floor((Math.sqrt(900 - row * row) / 5));  // 计算并存储平方根值

      // 开始循环遍历所有列
      for (column = root; column >= -root; column += -5) {

        plotVal = 25 + (int) Math.floor(func(Math.sqrt(row * row + column * column)) - 0.7 * column);  // 计算并存储绘图数值

        if (plotVal > limit) {  // 如果绘图数值大于限制值，则执行以下操作

          limit = plotVal;  // 更新限制值
          // Add whitespace
          // 在行内容的末尾添加空格，直到达到指定的列宽度
          while (lineContent.length() < (plotVal-1)) {
            lineContent += " ";
          }

          // 在行内容的末尾添加"*"，表示绘制游戏图形的一部分
          lineContent += "*";

        }

      }  // End loop through all columns

      // 打印出当前行的内容，表示绘制游戏图形的一行
      System.out.println(lineContent);

      // 重置行内容为空，准备绘制下一行
      lineContent = "";

    }  // End loop through all rows

  }  // End of method startGame
  // 定义一个函数，用于计算输入值的函数值
  public double func(double inputVal) {
    // 返回函数值
    return (30 * Math.exp(-inputVal * inputVal / 100));
  }

  // 主方法
  public static void main(String[] args) {
    // 创建一个3D绘图对象
    Plot3D plot = new Plot3D();
    // 调用绘图对象的play方法
    plot.play();
  }  // main方法结束

}  // Plot3D类结束
```