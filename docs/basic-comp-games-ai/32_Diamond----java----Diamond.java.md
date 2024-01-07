# `basic-computer-games\32_Diamond\java\Diamond.java`

```

import java.util.Scanner;

/**
 * Game of Diamond
 * <p>
 * Based on the BASIC game of Diamond here
 * https://github.com/coding-horror/basic-computer-games/blob/main/32%20Diamond/diamond.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

public class Diamond {

  private static final int LINE_WIDTH = 60;  // 定义屏幕宽度

  private static final String PREFIX = "CC";  // 定义前缀字符串

  private static final char SYMBOL = '!';  // 定义符号

  private final Scanner scan;  // 用于用户输入

  // 构造函数
  public Diamond() {
    scan = new Scanner(System.in);  // 初始化 Scanner 对象
  }  // End of constructor Diamond

  // 游戏开始
  public void play() {
    showIntro();  // 显示游戏介绍
    startGame();  // 开始游戏
  }  // End of method play

  // 显示游戏介绍
  private void showIntro() {
    System.out.println(" ".repeat(32) + "DIAMOND");  // 打印游戏标题
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印创意计算公司信息
    System.out.println("\n\n");  // 打印空行
  }  // End of method showIntro

  // 开始游戏
  private void startGame() {
    // 定义变量
    int body = 0;
    int column = 0;
    int end = 0;
    int fill = 0;
    int increment = 2;
    int numPerSide = 0;
    int prefixIndex = 0;
    int row = 0;
    int start = 1;
    int userNum = 0;
    String lineContent = "";

    // 获取用户输入
    System.out.println("FOR A PRETTY DIAMOND PATTERN,");
    System.out.print("TYPE IN AN ODD NUMBER BETWEEN 5 AND 21? ");
    userNum = scan.nextInt();  // 读取用户输入的奇数
    System.out.println("");

    // 计算每边要绘制的钻石数量
    numPerSide = (int) (LINE_WIDTH / userNum);

    end = userNum;

    // 循环绘制每一行的钻石
    for (row = 1; row <= numPerSide; row++) {
      // 循环绘制每个钻石的顶部和底部
      for (body = start; increment < 0 ? body >= end : body <= end; body += increment) {
        lineContent = "";

        // 添加空格
        while (lineContent.length() < ((userNum - body) / 2)) {
          lineContent += " ";
        }

        // 循环绘制每一列的钻石
        for (column = 1; column <= numPerSide; column++) {
          prefixIndex = 1;

          // 循环填充每个钻石的字符
          for (fill = 1; fill <= body; fill++) {
            // 钻石的右侧
            if (prefixIndex > PREFIX.length()) {
              lineContent += SYMBOL;
            }
            // 钻石的左侧
            else {
              lineContent += PREFIX.charAt(prefixIndex - 1);
              prefixIndex++;
            }
          }  // End loop that fills each diamond with characters

          // 列结束
          if (column == numPerSide) {
            break;
          }
          // 列未结束
          else {
            // 添加空格
            while (lineContent.length() < (userNum * column + (userNum - body) / 2)) {
              lineContent += " ";
            }
          }
        }  // End loop through each column of diamonds

        System.out.println(lineContent);  // 打印每行的钻石内容
      }  // End loop through top and bottom half of each diamond

      if (start != 1) {
        start = 1;
        end = userNum;
        increment = 2;
      }
      else {
        start = userNum - 2;
        end = 1;
        increment = -2;
        row--;
      }
    }  // End loop through each row of diamonds
  }  // End of method startGame

  // 主方法
  public static void main(String[] args) {
    Diamond diamond = new Diamond();
    diamond.play();  // 开始游戏
  }  // End of method main
}  // End of class Diamond

```