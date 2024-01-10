# `basic-computer-games\67_One_Check\java\OneCheck.java`

```
import java.util.Arrays;  // 导入 Arrays 类，用于操作数组
import java.util.Scanner;  // 导入 Scanner 类，用于接收用户输入

/**
 * Game of One Check
 * <p>
 * 基于 BASIC 游戏 One Check，原始代码在这里
 * https://github.com/coding-horror/basic-computer-games/blob/main/67%20One%20Check/onecheck.bas
 * <p>
 * 注意：这个想法是在 Java 中创建一个 1970 年代 BASIC 游戏的版本，没有引入新功能 - 没有添加额外的文本、错误检查等。
 *
 * 由 Darren Cardenas 从 BASIC 转换为 Java。
 */

public class OneCheck {

  private final Scanner scan;  // 用于用户输入的 Scanner 对象

  private enum Step {
    SHOW_INSTRUCTIONS, SHOW_BOARD, GET_MOVE, GET_SUMMARY, QUERY_RETRY  // 游戏进行的步骤
  }

  public OneCheck() {

    scan = new Scanner(System.in);  // 初始化 Scanner 对象

  }  // End of constructor OneCheck

  public void play() {

    showIntro();  // 显示游戏介绍
    startGame();  // 开始游戏

  }  // End of method play

  private static void showIntro() {

    System.out.println(" ".repeat(29) + "ONE CHECK");  // 打印游戏标题
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印创意计算机的信息
    System.out.println("\n\n");  // 打印空行

  }  // End of method showIntro

  private void startGame() {

    int fromSquare = 0;  // 起始方格
    int numJumps = 0;  // 跳跃次数
    int numPieces = 0;  // 棋子数量
    int square = 0;  // 方格
    int startPosition = 0;  // 起始位置
    int toSquare = 0;  // 目标方格

    // 移动合法性测试变量
    int fromTest1 = 0;  // 起始测试1
    int fromTest2 = 0;  // 起始测试2
    int toTest1 = 0;  // 目标测试1
    int toTest2 = 0;  // 目标测试2

    int[] positions = new int[65];  // 位置数组，用于存储方格状态

    Step nextStep = Step.SHOW_INSTRUCTIONS;  // 下一步骤为显示说明

    String lineContent = "";  // 行内容
    String userResponse = "";  // 用户响应

    // 开始外部 while 循环
    }  // End outer while loop

  }  // End of method startGame

  public void printBoard(int[] positions) {

    int column = 0;  // 列数
    int row = 0;  // 行数
    String lineContent = "";  // 行内容

    // 开始循环遍历所有行
    // 循环遍历行，每次增加8
    for (row = 1; row <= 57; row += 8) {

      // 开始循环遍历所有列
      for (column = row; column <= row + 7; column++) {

        // 将 positions 数组中的值添加到 lineContent 中
        lineContent += " " + positions[column];

      }  // 结束循环遍历所有列

      // 打印 lineContent
      System.out.println(lineContent);
      // 重置 lineContent 为空字符串
      lineContent = "";

    }  // 结束循环遍历所有行

    // 打印空行
    System.out.println("");

  }  // 结束 printBoard 方法

  // 主方法
  public static void main(String[] args) {

    // 创建 OneCheck 对象
    OneCheck game = new OneCheck();
    // 调用 play 方法
    game.play();

  }  // 结束 main 方法
# 类 OneCheck 的结束标记
```