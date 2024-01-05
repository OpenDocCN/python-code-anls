# `21_Calendar\java\Calendar.java`

```
/**
 * Game of Calendar
 * <p>
 * Based on the BASIC game of Calendar here
 * https://github.com/coding-horror/basic-computer-games/blob/main/21%20Calendar/calendar.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

public class Calendar {

  // 定义常量，表示每个月的周数和每周的天数
  private static final int NUM_WEEK_ROWS = 6;
  private static final int NUM_DAYS_PER_WEEK = 7;
  private static final int NUM_MONTHS_PER_YEAR = 12;
  // 定义数组，表示每个月的天数，索引0不使用
  private static final int[] daysPerMonth = { 0, 31, 28, 31, 30, 31, 30,
                                             31, 31, 30, 31, 30, 31 };

  public void play() {
    // 调用显示游戏介绍的方法
    showIntro();
    // 调用开始游戏的方法
    startGame();
  }  // End of method play

  // 显示游戏介绍的方法
  private static void showIntro() {
    // 打印游戏标题
    System.out.println(" ".repeat(31) + "CALENDAR");
    // 打印游戏信息
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");
  }  // End of method showIntro

  // 开始游戏的方法
  private void startGame() {
    // 初始化日期相关变量
    int dayOfMonth = 0;
    int dayOfWeek = 0;
    int dayOfYear = 0;
    # 初始化变量，用于存储总天数、索引、月份和行数
    daysTotal = 0;
    index = 0;
    month = 0;
    row = 0;

    # 初始化字符串变量，用于存储行内容
    lineContent = "";

    # 执行循环，打印空行，循环次数为6次
    for (index = 1; index <= 6; index++) {
      System.out.println("");
    }

    # 将总天数和一年中的天数初始化为负数
    daysTotal = -1;
    dayOfYear = 0;

    # 打印空行
    System.out.println("");

    # 开始循环遍历所有月份
    for (month = 1; month <= NUM_MONTHS_PER_YEAR; month++) {

      # 打印空行
      System.out.println("");
      dayOfYear = dayOfYear + daysPerMonth[month - 1];  // 将当月的天数加到年份的天数中

      lineContent = String.format("** %-3d" + "*".repeat(18), dayOfYear);  // 格式化输出当年的第几天，并用星号表示

      switch (month) {  // 根据月份进行不同的处理
        case 1:
          lineContent += " JANUARY ";  // 如果是1月，则在lineContent后面添加" JANUARY "
          break;
        case 2:
          lineContent += " FEBRUARY";  // 如果是2月，则在lineContent后面添加" FEBRUARY"
          break;
        case 3:
          lineContent += "  MARCH  ";  // 如果是3月，则在lineContent后面添加"  MARCH  "
          break;
        case 4:
          lineContent += "  APRIL  ";  // 如果是4月，则在lineContent后面添加"  APRIL  "
          break;
        case 5:
          lineContent += "   MAY   ";  // 如果是5月，则在lineContent后面添加"   MAY   "
          break;  // 结束当前的 case 分支，跳出 switch 语句
        case 6:  // 如果月份为 6，执行以下语句
          lineContent += "   JUNE  ";  // 在 lineContent 后面添加 "   JUNE  "
          break;  // 结束当前的 case 分支，跳出 switch 语句
        case 7:  // 如果月份为 7，执行以下语句
          lineContent += "   JULY  ";  // 在 lineContent 后面添加 "   JULY  "
          break;  // 结束当前的 case 分支，跳出 switch 语句
        case 8:  // 如果月份为 8，执行以下语句
          lineContent += "  AUGUST ";  // 在 lineContent 后面添加 "  AUGUST "
          break;  // 结束当前的 case 分支，跳出 switch 语句
        case 9:  // 如果月份为 9，执行以下语句
          lineContent += "SEPTEMBER";  // 在 lineContent 后面添加 "SEPTEMBER"
          break;  // 结束当前的 case 分支，跳出 switch 语句
        case 10:  // 如果月份为 10，执行以下语句
          lineContent += " OCTOBER ";  // 在 lineContent 后面添加 " OCTOBER "
          break;  // 结束当前的 case 分支，跳出 switch 语句
        case 11:  // 如果月份为 11，执行以下语句
          lineContent += " NOVEMBER";  // 在 lineContent 后面添加 " NOVEMBER"
          break;  // 结束当前的 case 分支，跳出 switch 语句
        case 12:  // 如果月份为 12，执行以下语句
      lineContent += " DECEMBER";  # 在lineContent字符串后面添加" DECEMBER"
      break;  # 跳出switch语句
    default:  # 默认情况
      break;  # 跳出switch语句
  }

  lineContent += "*".repeat(18) + " " + (365 - dayOfYear) + "**";  # 在lineContent字符串后面添加18个"*"，然后加上365 - dayOfYear的值，再加上两个"*"

  System.out.println(lineContent);  # 打印lineContent字符串
  System.out.println("");  # 打印空行

  System.out.print("     S       M       T       W");  # 打印星期的缩写
  System.out.println("       T       F       S");  # 打印星期的缩写

  System.out.println("");  # 打印空行

  System.out.println("*".repeat(59));  # 打印59个"*"

  // Begin loop through each week row  # 开始循环遍历每一周的行
  for (row = 1; row <= NUM_WEEK_ROWS; row++) {  # 循环条件，从1到NUM_WEEK_ROWS
        System.out.println(""); // 打印空行

        lineContent = "    "; // 初始化字符串变量lineContent为四个空格

        // 开始循环遍历一周的每一天
        for (dayOfWeek = 1; dayOfWeek <= NUM_DAYS_PER_WEEK; dayOfWeek++) {

          daysTotal++; // 总天数加一

          dayOfMonth = daysTotal - dayOfYear; // 计算当月的日期

          if (dayOfMonth > daysPerMonth[month]) { // 如果日期大于当月的天数
            row = 6; // 行数设为6
            break; // 跳出循环
          }

          if (dayOfMonth > 0) { // 如果日期大于0
            lineContent += dayOfMonth; // 将日期添加到字符串lineContent中
          }
          while (lineContent.length() < (4 + 8 * dayOfWeek)) {
            lineContent += " ";
          }
          // 在行内容的末尾添加空格，直到达到指定的长度（4 + 8 * dayOfWeek）

        }  // End loop through days of the week
        // 结束对一周中每天的循环

        if (dayOfMonth == daysPerMonth[month]) {
          row = 6;
          daysTotal += dayOfWeek;
          System.out.println(lineContent);
          break;
        }
        // 如果是月底，更新行数和总天数，打印行内容，然后跳出循环

        System.out.println(lineContent);
        // 打印行内容

      }  // End loop through each week row
      // 结束对每周行的循环

      daysTotal -= dayOfWeek;
      // 减去当前周的天数

    }  // End loop through all months
    // 结束对所有月份的循环
    for (index = 1; index <= 6; index++) {
      System.out.println("");
    }
```
这段代码是一个 for 循环，用于循环执行打印空行的操作，循环条件是 index 小于等于 6，每次循环 index 值加 1。

```
  }  // End of method startGame
```
这是一个方法结束的注释，标识着 startGame 方法的结束。

```
  public static void main(String[] args) {

    Calendar game = new Calendar();
    game.play();

  }  // End of method main
```
这是 main 方法的结束注释，标识着 main 方法的结束。

```
}  // End of class Calendar
```
这是类 Calendar 的结束注释，标识着类的结束。
```