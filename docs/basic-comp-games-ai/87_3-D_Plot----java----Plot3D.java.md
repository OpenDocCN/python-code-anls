# `basic-computer-games\87_3-D_Plot\java\Plot3D.java`

```

// 导入 java.lang.Math 包，用于数学计算
import java.lang.Math;

/**
 * 3-D Plot 游戏
 * <p>
 * 基于 BASIC 游戏 3-D Plot，链接在这里
 * https://github.com/coding-horror/basic-computer-games/blob/main/87%203-D%20Plot/3dplot.bas
 * <p>
 * 注意：这个想法是在 Java 中创建一个 1970 年代 BASIC 游戏的版本，没有引入新功能 - 没有添加额外的文本、错误检查等。
 *
 * 由 Darren Cardenas 从 BASIC 转换为 Java。
 */

// Java 类名不能以字母开头，所以不能使用类名 3dplot
public class Plot3D {


  public void play() {

    showIntro(); // 调用显示游戏介绍的方法
    startGame(); // 调用开始游戏的方法

  }  // End of method play


  private void showIntro() {

    System.out.println(" ".repeat(31) + "3D PLOT"); // 打印游戏标题
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"); // 打印创意计算的信息
    System.out.println("\n\n\n"); // 打印空行

  }  // End of method showIntro


  private void startGame() {

    float row = 0; // 初始化行数
    int column = 0; // 初始化列数
    int limit = 0; // 初始化限制值
    int plotVal = 0; // 初始化绘图值
    int root = 0; // 初始化根值

    String lineContent = ""; // 初始化行内容字符串

    // 开始循环遍历所有行
    for (row = -30; row <= 30; row += 1.5) {

      limit = 0; // 重置限制值

      root = 5 * (int) Math.floor((Math.sqrt(900 - row * row) / 5)); // 计算根值

      // 开始循环遍历所有列
      for (column = root; column >= -root; column += -5) {

        plotVal = 25 + (int) Math.floor(func(Math.sqrt(row * row + column * column)) - 0.7 * column); // 计算绘图值

        if (plotVal > limit) {

          limit = plotVal; // 更新限制值

          // 添加空格
          while (lineContent.length() < (plotVal-1)) {
            lineContent += " ";
          }

          lineContent += "*"; // 添加星号

        }

      }  // End loop through all columns

      System.out.println(lineContent); // 打印行内容

      lineContent = ""; // 重置行内容字符串

    }  // End loop through all rows

  }  // End of method startGame


  // 要绘制的函数
  public double func(double inputVal) {

    return (30 * Math.exp(-inputVal * inputVal / 100)); // 返回计算结果

  }


  public static void main(String[] args) {

    Plot3D plot = new Plot3D(); // 创建 Plot3D 对象
    plot.play(); // 调用 play 方法开始游戏

  }  // End of method main

}  // End of class Plot3D

```