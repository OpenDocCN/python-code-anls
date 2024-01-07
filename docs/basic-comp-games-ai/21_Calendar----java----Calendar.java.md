# `basic-computer-games\21_Calendar\java\Calendar.java`

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

  // 定义常量
  private static final int NUM_WEEK_ROWS = 6;
  private static final int NUM_DAYS_PER_WEEK = 7;
  private static final int NUM_MONTHS_PER_YEAR = 12;
  private static final int[] daysPerMonth = { 0, 31, 28, 31, 30, 31, 30,
                                             31, 31, 30, 31, 30, 31 };

  // 游戏开始方法
  public void play() {

    // 显示游戏介绍
    showIntro();
    // 开始游戏
    startGame();

  }  // End of method play

  // 显示游戏介绍
  private static void showIntro() {

    System.out.println(" ".repeat(31) + "CALENDAR");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");

  }  // End of method showIntro

  // 开始游戏
  private void startGame() {

    // 初始化变量
    int dayOfMonth = 0;
    int dayOfWeek = 0;
    int dayOfYear = 0;
    int daysTotal = 0;
    int index = 0;
    int month = 0;
    int row = 0;

    String lineContent = "";

    // 打印空行
    for (index = 1; index <= 6; index++) {
      System.out.println("");
    }

    daysTotal = -1;
    dayOfYear = 0;

    System.out.println("");

    // 开始循环遍历所有月份
    for (month = 1; month <= NUM_MONTHS_PER_YEAR; month++) {

      System.out.println("");

      dayOfYear = dayOfYear + daysPerMonth[month - 1];

      lineContent = String.format("** %-3d" + "*".repeat(18), dayOfYear);

      // 根据月份设置行内容
      switch (month) {
        case 1:
          lineContent += " JANUARY ";
          break;
        case 2:
          lineContent += " FEBRUARY";
          break;
        case 3:
          lineContent += "  MARCH  ";
          break;
        case 4:
          lineContent += "  APRIL  ";
          break;
        case 5:
          lineContent += "   MAY   ";
          break;
        case 6:
          lineContent += "   JUNE  ";
          break;
        case 7:
          lineContent += "   JULY  ";
          break;
        case 8:
          lineContent += "  AUGUST ";
          break;
        case 9:
          lineContent += "SEPTEMBER";
          break;
        case 10:
          lineContent += " OCTOBER ";
          break;
        case 11:
          lineContent += " NOVEMBER";
          break;
        case 12:
          lineContent += " DECEMBER";
          break;
        default:
          break;
      }

      lineContent += "*".repeat(18) + " " + (365 - dayOfYear) + "**";

      System.out.println(lineContent);
      System.out.println("");

      System.out.print("     S       M       T       W");
      System.out.println("       T       F       S");
      System.out.println("");

      System.out.println("*".repeat(59));

      // 开始循环遍历每一周
      for (row = 1; row <= NUM_WEEK_ROWS; row++) {

        System.out.println("");

        lineContent = "    ";

        // 开始循环遍历每一天
        for (dayOfWeek = 1; dayOfWeek <= NUM_DAYS_PER_WEEK; dayOfWeek++) {

          daysTotal++;

          dayOfMonth = daysTotal - dayOfYear;

          if (dayOfMonth > daysPerMonth[month]) {
            row = 6;
            break;
          }

          if (dayOfMonth > 0) {
            lineContent += dayOfMonth;
          }

          while (lineContent.length() < (4 + 8 * dayOfWeek)) {
            lineContent += " ";
          }

        }  // End loop through days of the week

        if (dayOfMonth == daysPerMonth[month]) {
          row = 6;
          daysTotal += dayOfWeek;
          System.out.println(lineContent);
          break;
        }

        System.out.println(lineContent);

      }  // End loop through each week row

      daysTotal -= dayOfWeek;

    }  // End loop through all months

    // 打印空行
    for (index = 1; index <= 6; index++) {
      System.out.println("");
    }

  }  // End of method startGame

  // 主方法
  public static void main(String[] args) {

    Calendar game = new Calendar();
    game.play();

  }  // End of method main

}  // End of class Calendar
```