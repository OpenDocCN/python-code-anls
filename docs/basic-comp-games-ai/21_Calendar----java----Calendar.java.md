# `basic-computer-games\21_Calendar\java\Calendar.java`

```py
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

  private static final int NUM_WEEK_ROWS = 6;  // 定义每个月的行数
  private static final int NUM_DAYS_PER_WEEK = 7;  // 定义每周的天数
  private static final int NUM_MONTHS_PER_YEAR = 12;  // 定义每年的月数
  private static final int[] daysPerMonth = { 0, 31, 28, 31, 30, 31, 30,
                                             31, 31, 30, 31, 30, 31 };  // 定义每个月的天数

  public void play() {

    showIntro();  // 调用显示游戏介绍的方法
    startGame();  // 调用开始游戏的方法

  }  // End of method play

  private static void showIntro() {

    System.out.println(" ".repeat(31) + "CALENDAR");  // 打印游戏标题
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");  // 打印游戏信息
    System.out.println("\n\n");  // 打印空行

  }  // End of method showIntro

  private void startGame() {

    int dayOfMonth = 0;  // 初始化每月的天数
    int dayOfWeek = 0;  // 初始化每周的天数
    int dayOfYear = 0;  // 初始化每年的天数
    int daysTotal = 0;  // 初始化总天数
    int index = 0;  // 初始化索引
    int month = 0;  // 初始化月份
    int row = 0;  // 初始化行数

    String lineContent = "";  // 初始化行内容

    for (index = 1; index <= 6; index++) {
      System.out.println("");  // 打印空行
    }

    daysTotal = -1;  // 初始化总天数为-1
    dayOfYear = 0;  // 初始化每年的天数为0

    System.out.println("");  // 打印空行

    // Begin loop through all months
    }  // End loop through all months

    for (index = 1; index <= 6; index++) {
      System.out.println("");  // 打印空行
    }

  }  // End of method startGame

  public static void main(String[] args) {

    Calendar game = new Calendar();  // 创建 Calendar 对象
    game.play();  // 调用 play 方法开始游戏

  }  // End of method main

}  // End of class Calendar
```