# `basic-computer-games\87_3-D_Plot\java\Plot3D.java`

```
import java.lang.Math;

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
public class Plot3D {


  public void play() {

    showIntro();
    startGame();

  }  // End of method play


  private void showIntro() {

    // 打印游戏介绍
    System.out.println(" ".repeat(31) + "3D PLOT");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n\n");

  }  // End of method showIntro


  private void startGame() {

    float row = 0;
    int column = 0;
    int limit = 0;
    int plotVal = 0;
    int root = 0;

    String lineContent = "";

    // Begin loop through all rows
    for (row = -30; row <= 30; row += 1.5) {

      limit = 0;

      root = 5 * (int) Math.floor((Math.sqrt(900 - row * row) / 5));

      // Begin loop through all columns
      for (column = root; column >= -root; column += -5) {

        plotVal = 25 + (int) Math.floor(func(Math.sqrt(row * row + column * column)) - 0.7 * column);

        if (plotVal > limit) {

          limit = plotVal;

          // Add whitespace
          while (lineContent.length() < (plotVal-1)) {
            lineContent += " ";
          }

          lineContent += "*";

        }

      }  // End loop through all columns

      // 打印每一行的内容
      System.out.println(lineContent);

      lineContent = "";

    }  // End loop through all rows

  }  // End of method startGame


  // Function to be plotted
  // 定义要绘制的函数
  public double func(double inputVal) {

    return (30 * Math.exp(-inputVal * inputVal / 100));

  }


  public static void main(String[] args) {
    // 创建一个名为plot的3D绘图对象
    Plot3D plot = new Plot3D();
    // 调用plot对象的play方法，开始播放3D绘图
    plot.play();

  }  // End of method main
# 结束 Plot3D 类的定义
}  // End of class Plot3D
```