# `d:/src/tocomm/basic-computer-games\32_Diamond\java\Diamond.java`

```
import java.util.Scanner;  # 导入 java.util.Scanner 包

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

  private static final int LINE_WIDTH = 60;  # 定义常量 LINE_WIDTH，值为 60

  private static final String PREFIX = "CC";  # 定义常量 PREFIX，值为 "CC"
  private static final char SYMBOL = '!';  // 定义一个常量 SYMBOL，值为 '!'，表示游戏中的特殊符号

  private final Scanner scan;  // 用于用户输入的 Scanner 对象

  public Diamond() {
    scan = new Scanner(System.in);  // 初始化 Scanner 对象，用于从控制台获取用户输入
  }  // Diamond 类的构造方法结束

  public void play() {
    showIntro();  // 调用 showIntro 方法，显示游戏介绍
    startGame();  // 调用 startGame 方法，开始游戏
  }  // play 方法结束
  private void showIntro() {

    // 打印游戏标题
    System.out.println(" ".repeat(32) + "DIAMOND");
    // 打印游戏信息
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    // 打印空行
    System.out.println("\n\n");

  }  // End of method showIntro


  private void startGame() {

    // 初始化变量
    int body = 0;
    int column = 0;
    int end = 0;
    int fill = 0;
    int increment = 2;
    int numPerSide = 0;
    int prefixIndex = 0;
    int row = 0;
    int start = 1;
    int userNum = 0; // 声明并初始化一个整型变量userNum，用于存储用户输入的数字

    String lineContent = ""; // 声明一个空字符串变量lineContent，用于存储每行的内容

    // 获取用户输入
    System.out.println("FOR A PRETTY DIAMOND PATTERN,");
    System.out.print("TYPE IN AN ODD NUMBER BETWEEN 5 AND 21? ");
    userNum = scan.nextInt(); // 从用户输入中获取一个整数并存储在userNum中
    System.out.println("");

    // 计算每边屏幕上要绘制的菱形数量
    numPerSide = (int) (LINE_WIDTH / userNum); // 计算每边屏幕上要绘制的菱形数量并存储在numPerSide中

    end = userNum; // 将end设置为用户输入的数字

    // 开始循环遍历每一行的菱形
    for (row = 1; row <= numPerSide; row++) {

      // 开始循环遍历每个菱形的顶部和底部
      for (body = start; increment < 0 ? body >= end : body <= end; body += increment) {
        lineContent = "";  // 初始化变量lineContent为空字符串

        // 添加空格
        while (lineContent.length() < ((userNum - body) / 2)) {  // 当lineContent的长度小于((userNum - body) / 2)时，执行循环
          lineContent += " ";  // 在lineContent末尾添加空格
        }

        // 开始循环遍历每个菱形的列
        for (column = 1; column <= numPerSide; column++) {  // 循环遍历每个菱形的列，从1到numPerSide

          prefixIndex = 1;  // 初始化变量prefixIndex为1

          // 开始循环填充每个菱形的字符
          for (fill = 1; fill <= body; fill++) {  // 循环填充每个菱形的字符，从1到body

            // 菱形的右侧
            if (prefixIndex > PREFIX.length()) {  // 如果prefixIndex大于PREFIX的长度时，执行以下操作

              lineContent += SYMBOL;  // 在lineContent末尾添加SYMBOL
            }
            // 如果不是菱形的左侧，即为右侧
            else {
              // 将字符添加到当前行的内容中
              lineContent += PREFIX.charAt(prefixIndex - 1);
              // 增加前缀索引
              prefixIndex++;
            }
          }  // 结束填充每个菱形的循环

          // 当前列完成
          if (column == numPerSide) {
            // 退出循环
            break;
          }
          // 当前列未完成
          else {
// 添加空格
while (lineContent.length() < (userNum * column + (userNum - body) / 2)) {
  lineContent += " ";
}
```
这段代码是在循环中，用空格填充lineContent字符串，直到它的长度达到指定的条件。

```
}  // End loop through each column of diamonds
```
这是对循环的结束进行注释，说明这一行是对前面循环的结束进行标记。

```
System.out.println(lineContent);
```
这行代码是打印lineContent字符串的内容。

```
}  // End loop through top and bottom half of each diamond
```
这是对循环的结束进行注释，说明这一行是对前面循环的结束进行标记。

```
if (start != 1) {
  start = 1;
  end = userNum;
  increment = 2;
```
这段代码是一个条件语句，如果start不等于1，则将start设置为1，end设置为userNum，increment设置为2。
      }
      else {
        # 如果用户输入的数字不是偶数，则将起始值设为用户输入值减去2
        start = userNum - 2;
        # 结束值设为1
        end = 1;
        # 步长设为-2
        increment = -2;
        # 行数减一
        row--;
      }
    }  // End loop through each row of diamonds
  }  // End of method startGame

  public static void main(String[] args) {
    # 创建 Diamond 对象
    Diamond diamond = new Diamond();
    # 调用 play 方法开始游戏
    diamond.play();
  }  // End of method main  // 结束 main 方法

}  // End of class Diamond  // 结束 Diamond 类
```