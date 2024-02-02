# `basic-computer-games\32_Diamond\java\Diamond.java`

```py
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

  private static final int LINE_WIDTH = 60;

  private static final String PREFIX = "CC";

  private static final char SYMBOL = '!';

  private final Scanner scan;  // For user input


  public Diamond() {

    scan = new Scanner(System.in);

  }  // End of constructor Diamond


  public void play() {

    showIntro();
    startGame();

  }  // End of method play


  private void showIntro() {

    System.out.println(" ".repeat(32) + "DIAMOND");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");

  }  // End of method showIntro


  private void startGame() {

    int body = 0;  // Initialize variable to store the body of the diamond
    int column = 0;  // Initialize variable to store the current column
    int end = 0;  // Initialize variable to store the end value
    int fill = 0;  // Initialize variable to store the fill value
    int increment = 2;  // Initialize variable to store the increment value
    int numPerSide = 0;  // Initialize variable to store the number of diamonds to be drawn on each side of the screen
    int prefixIndex = 0;  // Initialize variable to store the index of the prefix
    int row = 0;  // Initialize variable to store the current row
    int start = 1;  // Initialize variable to store the start value
    int userNum = 0;  // Initialize variable to store the user input

    String lineContent = "";  // Initialize variable to store the content of each line

    // Get user input
    System.out.println("FOR A PRETTY DIAMOND PATTERN,");  // Prompt the user for input
    System.out.print("TYPE IN AN ODD NUMBER BETWEEN 5 AND 21? ");  // Prompt the user for input
    userNum = scan.nextInt();  // Read user input
    System.out.println("");  // Print an empty line

    // Calcuate number of diamonds to be drawn on each side of screen
    numPerSide = (int) (LINE_WIDTH / userNum);  // Calculate the number of diamonds to be drawn on each side of the screen

    end = userNum;  // Set the end value to the user input

    // Begin loop through each row of diamonds
    for (row = 1; row <= numPerSide; row++) {
        // 循环遍历每一行的菱形
        // Begin loop through top and bottom halves of each diamond
        for (body = start; increment < 0 ? body >= end : body <= end; body += increment) {
            // 循环遍历每个菱形的顶部和底部
            lineContent = "";
            // 初始化每行的内容为空字符串

            // Add whitespace
            while (lineContent.length() < ((userNum - body) / 2)) {
                lineContent += " ";
            }
            // 添加空格，使得菱形居中显示

            // Begin loop through each column of diamonds
            for (column = 1; column <= numPerSide; column++) {
                // 循环遍历每个菱形的列
                prefixIndex = 1;
                // 初始化前缀索引为1

                // Begin loop that fills each diamond with characters
                for (fill = 1; fill <= body; fill++) {
                    // 循环填充每个菱形的字符

                    // Right side of diamond
                    if (prefixIndex > PREFIX.length()) {
                        // 如果前缀索引超出了前缀字符串的长度
                        lineContent += SYMBOL;
                        // 在行内容中添加符号
                    }
                    // Left side of diamond
                    else {
                        lineContent += PREFIX.charAt(prefixIndex - 1);
                        // 在行内容中添加前缀字符串中对应索引的字符
                        prefixIndex++;
                        // 前缀索引加一
                    }
                }  // End loop that fills each diamond with characters

                // Column finished
                if (column == numPerSide) {
                    // 如果列数等于每边菱形的数量
                    break;
                    // 跳出循环
                }
                // Column not finishd
                else {
                    // 如果列数不等于每边菱形的数量
                    // Add whitespace
                    while (lineContent.length() < (userNum * column + (userNum - body) / 2)) {
                        lineContent += " ";
                    }
                    // 添加空格，使得菱形居中显示
                }
            }  // End loop through each column of diamonds

            System.out.println(lineContent);
            // 打印每行的内容
        }  // End loop through top and bottom half of each diamond

        if (start != 1) {
            // 如果起始值不等于1
            start = 1;
            end = userNum;
            increment = 2;
            // 重新设置起始值、结束值和增量
        }
        else {
            // 如果起始值等于1
            start = userNum - 2;
            end = 1;
            increment = -2;
            row--;
            // 重新设置起始值、结束值和增量，并减少行数
        }
    }  // End loop through each row of diamonds
    // 循环遍历每一行的菱形结束

  }  // End of method startGame
  // startGame 方法结束


  public static void main(String[] args) {
    // 主方法
    Diamond diamond = new Diamond();
    diamond.play();
    // 创建 Diamond 对象并调用 play 方法
  }  // End of method main
  // 主方法结束
}  // Diamond 类的结束
```