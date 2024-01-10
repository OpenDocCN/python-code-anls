# `basic-computer-games\95_Weekday\java\Weekday.java`

```
import java.util.Scanner;

/**
 * WEEKDAY
 *
 * Converted from BASIC to Java by Aldrin Misquitta (@aldrinm)
 *
 */
public class Weekday {

    //TABLE OF VALUES FOR THE MONTHS TO BE USED IN CALCULATIONS.
    //Dummy value added at index 0, so we can reference directly by the month number value
    private final static int[] t = new int[]{-1, 0, 3, 3, 6, 1, 4, 6, 2, 5, 0, 3, 5};

    }


    private static void printStatisticRow(float F, int A8, DateStruct scratchDate) {
        int K1 = (int) (F * A8);  // Calculate K1 based on F and A8
        int I5 = K1 / 365;  // Calculate I5 based on K1
        K1 = K1 - (I5 * 365);  // Update K1
        int I6 = K1 / 30;  // Calculate I6 based on K1
        int I7 = K1 - (I6 * 30);  // Calculate I7 based on K1
        int K5 = scratchDate.year - I5;  // Calculate K5 based on scratchDate.year and I5
        int K6 = scratchDate.month - I6;  // Calculate K6 based on scratchDate.month and I6
        int K7 = scratchDate.day - I7;  // Calculate K7 based on scratchDate.day and I7
        if (K7 < 0) {  // Check if K7 is less than 0
            K7 = K7 + 30;  // Update K7
            K6 = K6 - 1;  // Update K6
        }
        if (K6 <= 0) {  // Check if K6 is less than or equal to 0
            K6 = K6 + 12;  // Update K6
            K5 = K5 - 1;  // Update K5
        }
        //to return the updated values of K5, K6, K7 we send them through the scratchDate
        scratchDate.year = K5;  // Update scratchDate.year
        scratchDate.month = K6;  // Update scratchDate.month
        scratchDate.day = K7;  // Update scratchDate.day
        System.out.printf("%14s%14s%14s%n", I5, I6, I7);  // Print the values of I5, I6, I7
    }

    private static void printHeadersAndAge(int I5, int I6, int I7) {
        System.out.printf("%14s%14s%14s%14s%14s%n", " ", " ", "YEARS", "MONTHS", "DAYS");  // Print the headers for the age
        System.out.printf("%14s%14s%14s%14s%14s%n", " ", " ", "-----", "------", "----");  // Print the separator line
        System.out.printf("%-28s%14s%14s%14s%n", "YOUR AGE (IF BIRTHDATE)", I5, I6, I7);  // Print the age based on I5, I6, I7
    }
    # 计算并打印日期所在的星期几
    private static void calculateAndPrintDayOfWeek(int i1, int a, DateStruct dateStruct, DateStruct dateOfInterest, int y3) {
        # 计算 b 的值
        int b = (a - b(a) * 7) + 1;
        # 如果日期的月份大于 2，则打印日期所在的星期几
        if (dateOfInterest.month > 2) {
            printDayOfWeek(dateStruct, dateOfInterest, b);
        } else {
            # 如果 y3 等于 0
            if (y3 == 0) {
                # 计算 aa 和 t1 的值
                int aa = i1 - 1;
                int t1 = aa - a(aa) * 4;
                # 如果 t1 等于 0
                if (t1 == 0) {
                    # 如果 b 不等于 0，则 b 减 1 并打印日期所在的星期几
                    if (b != 0) {
                        b = b - 1;
                        printDayOfWeek(dateStruct, dateOfInterest, b);
                    } else {
                        # 否则，b 等于 6，再减 1 并打印日期所在的星期几
                        b = 6;
                        b = b - 1;
                        printDayOfWeek(dateStruct, dateOfInterest, b);
                    }
                }
            }
        }
    }

    /**
     * 打印日期所在的星期几
     */
    // 打印给定日期是星期几
    private static void printDayOfWeek(DateStruct dateStruct, DateStruct dateOfInterest, int b) {
        // 如果 b 等于 0，则将其赋值为 7
        if (b == 0) {
            b = 7;
        }
        // 判断给定日期是否在感兴趣日期之前，然后打印相应信息
        if ((dateStruct.year * 12 + dateStruct.month) * 31
                + dateStruct.day
                <
                (dateOfInterest.year * 12
                        + dateOfInterest.month) * 31 + dateOfInterest.day) {
            System.out.printf("%s / %s / %s WILL BE A ", dateOfInterest.month, dateOfInterest.day, dateOfInterest.year);
        } else if ((dateStruct.year * 12 + dateStruct.month) * 31
                + dateStruct.day == (dateOfInterest.year * 12 + dateOfInterest.month)
                * 31 + dateOfInterest.day) {
            System.out.printf("%s / %s / %s IS A ", dateOfInterest.month, dateOfInterest.day, dateOfInterest.year);
        } else {
            System.out.printf("%s / %s / %s WAS A ", dateOfInterest.month, dateOfInterest.day, dateOfInterest.year);
        }
        // 根据 b 的值打印对应的星期几
        switch (b) {
            case 1:
                System.out.println("SUNDAY.");
                break;
            case 2:
                System.out.println("MONDAY.");
                break;
            case 3:
                System.out.println("TUESDAY.");
                break;
            case 4:
                System.out.println("WEDNESDAY.");
                break;
            case 5:
                System.out.println("THURSDAY.");
                break;
            case 6:
                // 如果是星期五并且日期是13号，则打印特殊信息，否则打印普通信息
                if (dateOfInterest.day == 13) {
                    System.out.println("FRIDAY THE THIRTEENTH---BEWARE!");
                } else {
                    System.out.println("FRIDAY.");
                }
                break;
            case 7:
                System.out.println("SATURDAY.");
                break;
        }
    }

    // 计算 a 除以 4 的结果
    private static int a(int a) {
        return a / 4;
    }

    // 计算 a 除以 7 的结果
    private static int b(int a) {
        return a / 7;
    }
    # 打印程序介绍信息
    private static void printIntro() {
        System.out.println("                                WEEKDAY");
        System.out.println("              CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println("\n\n\n");
        System.out.println("WEEKDAY IS A COMPUTER DEMONSTRATION THAT");
        System.out.println("GIVES FACTS ABOUT A DATE OF INTEREST TO YOU.");
        System.out.println("\n");
    }

    /**
     * 读取用户输入的日期，进行一些验证并返回一个简单的日期结构
     */
    private static DateStruct readDate(Scanner scanner) {
        boolean done = false;
        int mm = 0, dd = 0, yyyy = 0;
        while (!done) {
            # 读取用户输入
            String input = scanner.next();
            # 以逗号为分隔符拆分输入
            String[] tokens = input.split(",");
            # 验证输入格式
            if (tokens.length < 3) {
                System.out.println("DATE EXPECTED IN FORM: 3,24,1979 - RETRY INPUT LINE");
            } else {
                try {
                    # 将拆分后的字符串转换为整数
                    mm = Integer.parseInt(tokens[0]);
                    dd = Integer.parseInt(tokens[1]);
                    yyyy = Integer.parseInt(tokens[2]);
                    done = true;
                } catch (NumberFormatException nfe) {
                    System.out.println("NUMBER EXPECTED - RETRY INPUT LINE");
                }
            }
        }
        # 返回日期结构
        return new DateStruct(mm, dd, yyyy);
    }

    /**
     * 便利的日期结构，用于保存用户输入的日期
     */
    private static class DateStruct {
        int month;
        int day;
        int year;

        public DateStruct(int month, int day, int year) {
            this.month = month;
            this.day = day;
            this.year = year;
        }
    }
# 闭合前面的函数定义
```