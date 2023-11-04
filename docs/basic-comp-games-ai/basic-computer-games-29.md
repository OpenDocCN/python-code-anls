# BasicComputerGames源码解析 29

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


# `21_Calendar/java/Calendar.java`

This is a Java program that creates a game of curling. The curling game is played on a 26-foot long ice rink, with 4在人endZone and 8 betweenZone ends. The teams play against each other in a variety of different formats, such as scrimmage，冰盆地 (a type of round-robin tournament) and team against team. The curling game uses the stopwatch to keep track of the time for each shot and the list to keep track of the team that shot the stone.

The program starts by creating a new Calendar object and setting the initial position of the stone in the back of the house to 22-24-26. It then sets the stroke number to 5 and the炉温 to 100. It then loops through the different modes of the game and updates the score, resetting it after each game mode. It also uses a for loop to print out the current score and the updated score in the last 20 seconds.

It then creates a new loop to start the game and runs it for 15 minutes. After the game is finished, it prints out the final score.

The program is using several methods, including:

* `startGame()` method which starts the game with the initial position of the stone and the number of players in the house, and the different game modes that can be selected.
* `runGame()` method which controls the flow of the game, updating the score and resetting it after each game mode, and it also prints out the final score.
* `main(String[] args)` method which is the entry point for the program, it creates a new instance of the `Calendar` class and runs the `runGame()` method.
* `createCalendarObject()` method which creates a new `Calendar` object and sets the initial position of the stone in the back of the house to a specific distance and the stroke number and the炉温 to a specific temperature.
* `setStrokeNumber(int stroke)` method which sets the stroke number to a specific number.
* `setRecallNumber(int recall)` method which sets the recall number to a specific number.
* `setGameType(String gameType)` method which sets the game type to one of the different game modes.
* `printScore()` method which prints out the current score.
* `updateScore()` method which updates the score after a game mode.
* `sendScore()` method which sends the score to the other player.
* `sendGameType()` method which sends the game type to the other player.
* `setTime(int time)` method which sets the time for the shot.
* `setHouse(int house)` method which sets the house for the shot.
* `sendShot()` method which sends the shot to the other player.
* `setReminder(int reminder)` method which sets the reminder for the shot.


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

  private static final int NUM_WEEK_ROWS = 6;
  private static final int NUM_DAYS_PER_WEEK = 7;
  private static final int NUM_MONTHS_PER_YEAR = 12;
  private static final int[] daysPerMonth = { 0, 31, 28, 31, 30, 31, 30,
                                             31, 31, 30, 31, 30, 31 };

  public void play() {

    showIntro();
    startGame();

  }  // End of method play

  private static void showIntro() {

    System.out.println(" ".repeat(31) + "CALENDAR");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");

  }  // End of method showIntro

  private void startGame() {

    int dayOfMonth = 0;
    int dayOfWeek = 0;
    int dayOfYear = 0;
    int daysTotal = 0;
    int index = 0;
    int month = 0;
    int row = 0;

    String lineContent = "";

    for (index = 1; index <= 6; index++) {
      System.out.println("");
    }

    daysTotal = -1;
    dayOfYear = 0;

    System.out.println("");

    // Begin loop through all months
    for (month = 1; month <= NUM_MONTHS_PER_YEAR; month++) {

      System.out.println("");

      dayOfYear = dayOfYear + daysPerMonth[month - 1];

      lineContent = String.format("** %-3d" + "*".repeat(18), dayOfYear);

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

      // Begin loop through each week row
      for (row = 1; row <= NUM_WEEK_ROWS; row++) {

        System.out.println("");

        lineContent = "    ";

        // Begin loop through days of the week
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

    for (index = 1; index <= 6; index++) {
      System.out.println("");
    }

  }  // End of method startGame

  public static void main(String[] args) {

    Calendar game = new Calendar();
    game.play();

  }  // End of method main

}  // End of class Calendar

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `21_Calendar/javascript/calendar.js`

这段代码定义了两个函数：`print()` 和 `tab()`。

`print()` 函数的作用是接收一个字符串参数（`str`），将其输出到网页上的一个元素中。这个元素在网页上通过 `document.getElementById("output").appendChild(document.createTextNode(str))` 来动态添加。函数输出的字符串被作为文档的 `appendChild()` 方法的一部分传递给 `document.getElementById("output")`，这意味着该元素将被添加到当前的文档中。

`tab()` 函数的作用是生成一个字符串，其中包含指定数量的制表符（`space`）。这个字符串是通过在空格中插入空间数量来生成的，然后通过返回该字符串来返回结果。在函数内部，变量 `str` 被初始化为空字符串，然后通过 `while (space-- > 0)` 循环来在空格中插入空间数量（`space`），最后返回生成的字符串。


```
// CALENDAR
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//
function print(str)
{
	document.getElementById("output").appendChild(document.createTextNode(str));
}

function tab(space)
{
	var str = "";
	while (space-- > 0)
		str += " ";
	return str;
}

```

这段代码会输出一系列日历表格。第一行输出的是32号日历的表格，第二行输出的是15号日历的表格，第三行输出的是在新历法下的纽约市时间的日历表格，第四行到第七行输出的是1979年1月1日到6月30日的日历表格，共365天。数组变量m中存储了0到31号日历的日期。变量d用来记录1979年1月1日是星期几，初始值为-1。变量s用来记录 Creative Computing 公司位于新泽西州的商业日期。


```
print(tab(32) + "CALENDAR\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
print("\n");
print("\n");
print("\n");

//       0, 31, 29  ON LEAP YEARS
var m = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

// VALUES FOR 1979 - SEE NOTES
for (i = 1; i <= 6; i++)
	print("\n");

d = -1;	// 1979 starts on Monday (0 = Sun, -1 = Monday, -2 = Tuesday)
s = 0;

```

It appears that you are trying to generate a calendar that displays the full name of each month, rather than just the abbreviated name. In this case, the code you provided already does this. However, there is a small issue with the output formatting. When printing the full name of a month, the asterisks are being padded at the end, which means that the first few lines of the output will be indistinguishable from the print output. To fix this, you can remove the space between the asterisks and the month name, like this:

print("     " + str + "\n");

This will print the full name of the month followed by a space and then the print output, which should allow you to easily distinguish the first few lines of the output from the print output.

Alternatively, you can also print the month name with a space in between the asterisks, like this:

print(str + "\n")
print("     " + str + "\n")

This should also allow you to easily distinguish the first few lines of the output from the print output.

I hope this helps! Let me know if you have any other questions.



```
for (n = 1; n <= 12; n++) {
	print("\n");
	print("\n");
	s = s + m[n - 1];
	str = "**" + s;
	while (str.length < 7)
		str += " ";
	for (i = 1; i <= 18; i++)
		str += "*";
	switch (n) {
		case  1:	str += " JANUARY "; break;
		case  2:	str += " FEBRUARY"; break;
		case  3:	str += "  MARCH  "; break;
		case  4:	str += "  APRIL  "; break;
		case  5:	str += "   MAY   "; break;
		case  6:	str += "   JUNE  "; break;
		case  7:	str += "   JULY  "; break;
		case  8:	str += "  AUGUST "; break;
		case  9:	str += "SEPTEMBER"; break;
		case 10:	str += " OCTOBER "; break;
		case 11:	str += " NOVEMBER"; break;
		case 12:	str += " DECEMBER"; break;
	}
	for (i = 1; i <= 18; i++)
		str += "*";
	str += (365 - s) + "**";
	     // 366 - s on leap years
	print(str + "\n");
	print("     S       M       T       W       T       F       S\n");
	print("\n");
	str = "";
	for (i = 1; i <= 59; i++)
		str += "*";
	for (week = 1; week <= 6; week++) {
		print(str + "\n");
		str = "    ";
		for (g = 1; g <= 7; g++) {
			d++;
			d2 = d - s;
			if (d2 > m[n]) {
				week = 6;
				break;
			}
			if (d2 > 0)
				str += d2;
			while (str.length < 4 + 8 * g)
				str += " ";
		}
		if (d2 == m[n]) {
			d += g;
			break;
		}
	}
	d -= g;
	print(str + "\n");
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)

Actually, this is not so much a port as a complete rewrite, making use of
Perl's Posix time functionality. The calendar is for the current year (not
1979), but you can get another year by specifying it on the command line, e.g.

 `perl 21_Calendar/perl/calendar.pl 2001`

It *may* even produce output in languages other than English. But the
leftmost column will still be Sunday, even in locales where it is
typically Monday.


# `21_Calendar/python/calendar.py`

这段代码是一个BASIC程序，用于打印任何年份的日历。它包含两个主要函数：打印日历和平滑日历。以下是函数的详细解释：

1. `print_calendar`函数：

这个函数接收一个年份参数，然后使用特定于这个年份的修改函数来打印日历。具体来说，它使用`LEAP_DAY`函数来检查给定的年份是否是闰年，如果是，则使用`SECONDS_DAY`函数来找到1月1日是星期几。然后，它使用`WHILE`循环来打印日历。

2. `main`函数：

这个函数是主函数。它首先定义了输入的文件名，然后打开该文件并读取其内容。接下来，它读取用户输入的年份，然后调用`print_calendar`函数来打印相应的日历。

总的来说，这个程序的主要目的是提供一个方便的、易于阅读和理解的日历打印程序。


```
"""
Calendar

From: BASIC Computer Games (1978)
      Edited by David Ahl#

   This program prints out a calendar
for any year. You must specify the
starting day of the week of the year in
statement 130. (Sunday(0), Monday
(-1), Tuesday(-2), etc.) You can determine
this by using the program WEEKDAY.
You must also make two changes
for leap years in statement 360 and 620.
The program listing describes the necessary
```

这段代码是一个Python程序，它接受用户输入并输出一个12个月的日历。

程序内部定义了一个名为`parse_input`的函数，用于处理用户输入的 weekday 和 leap year 选项。

函数首先定义了一个`days_mapping`字典，用于存储特定日期的Weekday数字，例如，`"sunday"`对应数字0，`"monday"`对应数字-1，`"tuesday"`对应数字-2，以此类推。

然后，定义了一个`day`变量和一个`leap_day`变量，用于表示输入的 weekday 和 leap year 是否正确。

接着，定义了一个`correct_day_input`变量，用于存储输入的 weekday 是否正确。

然后，程序会循环地接受用户输入的 weekday 和 leap year，并将其存储在相应的变量中。

接下来，程序会检查输入的 weekday 和 leap year 是否正确。如果 weekday 正确，且输入的是 leap year，则程序会将`leap_day`设置为`True`，否则设置为`False`。

最后，程序会输出一个12个月的日历，包括闰年和普通年。


```
changes. Running the program produces a
nice 12-month calendar.
   The program was written by Geofrey
Chase of the Abbey, Portsmouth, Rhode Island.
"""

from typing import Tuple


def parse_input() -> Tuple[int, bool]:
    """
    function to parse input for weekday and leap year boolean
    """

    days_mapping = {
        "sunday": 0,
        "monday": -1,
        "tuesday": -2,
        "wednesday": -3,
        "thursday": -4,
        "friday": -5,
        "saturday": -6,
    }

    day = 0
    leap_day = False

    correct_day_input = False
    while not correct_day_input:
        weekday = input("INSERT THE STARTING DAY OF THE WEEK OF THE YEAR:")

        for day_k in days_mapping.keys():
            if weekday.lower() in day_k:
                day = days_mapping[day_k]
                correct_day_input = True
                break

    while True:
        leap = input("IS IT A LEAP YEAR?:")

        if "y" in leap.lower():
            leap_day = True
            break

        if "n" in leap.lower():
            leap_day = False
            break

    return day, leap_day


```

The given code appears to be a Python program that prints a table of days for a specific time period. It does this by first defining a list of days, a string for the separator between the days, and a variable for the number of days in a year.

It then enters a loop that runs through the days, printing each day and separating it with a space. After the loop, it loops through the days again and prints a separator line.

The program also includes a loop that advances the date by one day, taking into account any leap years. This is done by subtracting the number of days from the original number of days based on whether the year is a leap year.

The program also includes a loop that outputs the days count, which is simply the count of days in each month.

Overall, the program appears to be well-structured and easy to read.


```
def calendar(weekday: int, leap_year: bool) -> None:
    """
    function to print a year's calendar.

    input:
        _weekday_: int - the initial day of the week (0=SUN, -1=MON, -2=TUES...)
        _leap_year_: bool - indicates if the year is a leap year
    """
    months_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days = "S        M        T        W        T        F        S\n"
    sep = "*" * 59
    years_day = 365
    d = weekday

    if leap_year:
        months_days[2] = 29
        years_day = 366

    months_names = [
        " JANUARY ",
        " FEBRUARY",
        "  MARCH  ",
        "  APRIL  ",
        "   MAY   ",
        "   JUNE  ",
        "   JULY  ",
        "  AUGUST ",
        "SEPTEMBER",
        " OCTOBER ",
        " NOVEMBER",
        " DECEMBER",
    ]

    days_count = 0  # S in the original program

    # main loop
    for n in range(1, 13):
        days_count += months_days[n - 1]
        print(
            f"** {days_count} ****************** {months_names[n - 1]} "
            f"****************** {years_day - days_count} **\n"
        )
        print(days)
        print(sep)

        for _ in range(1, 7):
            print("\n")
            for g in range(1, 8):  # noqa
                d += 1
                d2 = d - days_count

                if d2 > months_days[n]:
                    break

                if d2 <= 0:
                    print("  ", end="       ")
                elif d2 < 10:
                    print(f" {d2}", end="       ")
                else:
                    print(f"{d2}", end="       ")
            print()

            if d2 >= months_days[n]:
                break

        if d2 > months_days[n]:
            d -= g

        print("\n")

    print("\n")


```

这段代码是一个Python程序，名为“main”。它定义了一个主函数（main function）和三个辅助函数（不输出源代码）。现在，我会解释这段代码的作用。

首先，我们需要了解这些函数的作用。

1. `print(" " * 32 + "CALENDAR")`：

这一行代码使用32个空格打印一个“CALENDAR”。32是2月份可能拥有的天数（平年2月份），所以打印出来的字符串是在告诉我们要计算2月份的日历。

2. `print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")`：

这一行代码同样使用15个空格打印一个字符串。15是圣诞节后的第一个星期几，所以打印出来的字符串是在告诉我们要计算圣诞节后的第一个星期的日历。

3. `print("\n" * 11)`：

这一行代码输出11个换行符，所以它在输出字符串时会分成11行输出。

接下来，我们来看看这些函数如何协作以实现整个程序的主要功能。

整个程序的主要目的是让用户输入平年和闰年的数量，然后输出相应的日历。所以，程序的逻辑分割为以下几个部分：

1. 解析用户输入：

`parse_input()`函数将用户输入的字符串解析为平年和闰年的数量。这个函数没有输出，因为它只在内部使用。

2. 调用日历函数：

`calendar(day, leap_year)`函数将在已知的平年和闰年数量的基础上计算出指定的日历。这个函数有输出，因为它会在主函数中打印结果。

3. 在主函数中打印结果：

`main()`函数是程序的入口点。它调用`calendar()`函数并打印结果。由于`print()`函数在内部使用了`print(" " * 32 + "CALENDAR")`，所以主函数的输出将是：“CALENDAR创造性计算Morristown, New Jersey 不断创新”。

然后，程序将输出一个换行符，并在打印日历后结束。


```
def main() -> None:
    print(" " * 32 + "CALENDAR")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    print("\n" * 11)

    day, leap_year = parse_input()
    calendar(day, leap_year)


if __name__ == "__main__":
    main()

########################################################
#
# Porting notes:
```

这段代码是一个简单的 Python 程序，它具有以下功能：

1. 在程序的开头添加了一个输入行，用于让用户指定一年的第一天以及该年是否为闰年。
2. 使用 `print()` 函数在屏幕上输出当前日期。
3. 包含一个多行注释，用于在代码中添加注释，方便阅读。
4. 使用了 Python 的 `pass` 语句，用于程序流程控制，确保程序只进入必要的循环。

该程序的主要目的是让用户输入一个年份，然后输出该年是否为闰年。


```
#
# It has been added an input at the beginning of the
# program so the user can specify the first day of the
# week of the year and if the year is leap or not.
#
########################################################

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by [Uğur Küpeli](https://github.com/ugurkupeli)

Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Change

In this program, the computer pretends it is the cashier at your friendly neighborhood candy store. You tell it the cost of the item(s) you are buying, the amount of your payment, and it will automatically (!) determine your correct change. Aren’t machines wonderful? Dennis Lunder of People’s Computer Company wrote this program.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=39)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=54)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `22_Change/csharp/Program.cs`

This is a C# program that sells the collected materials.

It provides a function called `GetInput()` that takes two arguments, a boolean indicating whether to accept or reject the input, and a double or int representing the amount to be paid.

This function is called repeatedly while the program is running, with the first call resulting in the user paying 10 cents, the second call resulting in the user paying 25 cents, and so on.

The program also includes a `PrintChange()` function that takes a double or int representing the change in the price of the collected materials, and prints it out in one of the following formats: "Sorry, you have short-changed me <percentage>!" or "Your change <amount>."

The program ends with a `Main()` function that sets the header text and a default amount to be sold, and continues running until the user is prompted to stop.


```
﻿using System;

namespace Change
{
    class Program
    {
        /// <summary>
        /// Prints header.
        /// </summary>
        static void Header()
        {
            Console.WriteLine("Change".PadLeft(33));
            Console.WriteLine("Creative Computing Morristown, New Jersey".PadLeft(15));
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("I, your friendly microcomputer, will determine\n"
            + "the correct change for items costing up to $100.");
            Console.WriteLine();
            Console.WriteLine();
        }

        /// <summary>
        /// Gets user input for price and payment.
        /// </summary>
        /// <returns>
        /// False if any input can't be parsed to double. Price and payment returned would be 0.
        /// True if it was possible to parse inputs into doubles. Price and payment returned
	    /// would be as provided by the user.
	    /// </returns>
        static (bool status, double price, double payment) GetInput()
        {
            Console.Write("Cost of item? ");
            var priceString = Console.ReadLine();
            if (!double.TryParse(priceString, out double price))
            {
                Console.WriteLine($"{priceString} isn't a number!");
                return (false, 0, 0);
            }

            Console.Write("Amount of payment? ");
            var paymentString = Console.ReadLine();
            if (!double.TryParse(paymentString, out double payment))
            {
                Console.WriteLine($"{paymentString} isn't a number!");
                return (false, 0, 0);
            }

            return (true, price, payment);
        }

        /// <summary>
        /// Prints bills and coins for given change.
        /// </summary>
        /// <param name="change"></param>
        static void PrintChange(double change)
        {
            var tens = (int)(change / 10);
            if (tens > 0)
                Console.WriteLine($"{tens} ten dollar bill(s)");

            var temp = change - (tens * 10);
            var fives = (int)(temp / 5);
            if (fives > 0)
                Console.WriteLine($"{fives} five dollar bill(s)");

            temp -= fives * 5;
            var ones = (int)temp;
            if (ones > 0)
                Console.WriteLine($"{ones} one dollar bill(s)");

            temp -= ones;
            var cents = temp * 100;
            var half = (int)(cents / 50);
            if (half > 0)
                Console.WriteLine($"{half} one half dollar(s)");

            temp = cents - (half * 50);
            var quarters = (int)(temp / 25);
            if (quarters > 0)
                Console.WriteLine($"{quarters} quarter(s)");

            temp -= quarters * 25;
            var dimes = (int)(temp / 10);
            if (dimes > 0)
                Console.WriteLine($"{dimes} dime(s)");

            temp -= dimes * 10;
            var nickels = (int)(temp / 5);
            if (nickels > 0)
                Console.WriteLine($"{nickels} nickel(s)");

            temp -= nickels * 5;
            var pennies = (int)(temp + 0.5);
            if (pennies > 0)
                Console.WriteLine($"{pennies} penny(s)");
        }

        static void Main(string[] args)
        {
            Header();

            while (true)
            {
                (bool result, double price, double payment) = GetInput();
                if (!result)
                    continue;

                var change = payment - price;
                if (change == 0)
                {
                    Console.WriteLine("Correct amount, thank you!");
                    continue;
                }

                if (change < 0)
                {
                    Console.WriteLine($"Sorry, you have short-changed me ${price - payment:N2}!");
                    continue;
                }

                Console.WriteLine($"Your change ${change:N2}");
                PrintChange(change);
                Console.WriteLine("Thank you, come again!");
                Console.WriteLine();
            }
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `22_Change/java/src/Change.java`

This appears to be a Java program that simulates a change of "$100" in a store by asking the player to enter the number of spaces between the desired amount and the actual amount.

It starts by defining a `Change` class with a method `displayTextAndGetNumber()` that takes a message to be displayed on the screen and returns the double value of the number entered by the player.

The `intro()` method is then used to print the name of the store, a header, and a message asking the player to enter the number of spaces between the desired amount and the actual amount.

The `simulateTabs()` method is used to simulate the appearance of spaces between the text, based on the number of spaces entered by the player.

The program also includes a `DisplayTextAndGetNumber()` method, which takes a message to be displayed on the screen and returns the double value of the number entered by the player, as well as a `displayTextAndGetInput()` method, which takes a message to be displayed on the screen and returns the key entered by the player.

It is not clear from this code snippet what the exact implementation is, but it appears to be setting up a system where the player can enter a number of spaces between the desired amount and the actual amount, and then displaying the correct change on the screen.


```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Change
 * <p>
 * Based on the Basic game of Change here
 * https://github.com/coding-horror/basic-computer-games/blob/main/22%20Change/change.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's Basic game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 */
public class Change {

    // Used for keyboard input
    private final Scanner kbScanner;

    private enum GAME_STATE {
        START_GAME,
        INPUT,
        CALCULATE,
        END_GAME,
        GAME_OVER
    }

    // Current game state
    private GAME_STATE gameState;

    // Amount of change needed to be given
    private double change;

    public Change() {
        kbScanner = new Scanner(System.in);

        gameState = GAME_STATE.START_GAME;
    }

    /**
     * Main game loop
     */
    public void play() {

        do {
            switch (gameState) {
                case START_GAME:
                    intro();
                    gameState = GAME_STATE.INPUT;
                    break;

                case INPUT:

                    double costOfItem = displayTextAndGetNumber("COST OF ITEM ");
                    double amountPaid = displayTextAndGetNumber("AMOUNT OF PAYMENT ");
                    change = amountPaid - costOfItem;
                    if (change == 0) {
                        // No change needed
                        System.out.println("CORRECT AMOUNT, THANK YOU.");
                        gameState = GAME_STATE.END_GAME;
                    } else if (change < 0) {
                        System.out.println("YOU HAVE SHORT-CHANGES ME $" + (costOfItem - amountPaid));
                        // Don't change game state so it will loop back and try again
                    } else {
                        // Change needed.
                        gameState = GAME_STATE.CALCULATE;
                    }
                    break;

                case CALCULATE:
                    System.out.println("YOUR CHANGE, $" + change);
                    calculateChange();
                    gameState = GAME_STATE.END_GAME;
                    break;

                case END_GAME:
                    System.out.println("THANK YOU, COME AGAIN");
                    System.out.println();
                    gameState = GAME_STATE.INPUT;
            }
        } while (gameState != GAME_STATE.GAME_OVER);
    }

    /**
     * Calculate and output the change required for the purchase based on
     * what money was paid.
     */
    private void calculateChange() {

        double originalChange = change;

        int tenDollarBills = (int) change / 10;
        if (tenDollarBills > 0) {
            System.out.println(tenDollarBills + " TEN DOLLAR BILL(S)");
        }
        change = originalChange - (tenDollarBills * 10);

        int fiveDollarBills = (int) change / 5;
        if (fiveDollarBills > 0) {
            System.out.println(fiveDollarBills + " FIVE DOLLAR BILL(S)");
        }
        change = originalChange - (tenDollarBills * 10 + fiveDollarBills * 5);

        int oneDollarBills = (int) change;
        if (oneDollarBills > 0) {
            System.out.println(oneDollarBills + " ONE DOLLAR BILL(S)");
        }
        change = originalChange - (tenDollarBills * 10 + fiveDollarBills * 5 + oneDollarBills);

        change = change * 100;
        double cents = change;

        int halfDollars = (int) change / 50;
        if (halfDollars > 0) {
            System.out.println(halfDollars + " ONE HALF DOLLAR(S)");
        }
        change = cents - (halfDollars * 50);

        int quarters = (int) change / 25;
        if (quarters > 0) {
            System.out.println(quarters + " QUARTER(S)");
        }

        change = cents - (halfDollars * 50 + quarters * 25);

        int dimes = (int) change / 10;
        if (dimes > 0) {
            System.out.println(dimes + " DIME(S)");
        }

        change = cents - (halfDollars * 50 + quarters * 25 + dimes * 10);

        int nickels = (int) change / 5;
        if (nickels > 0) {
            System.out.println(nickels + " NICKEL(S)");
        }

        change = cents - (halfDollars * 50 + quarters * 25 + dimes * 10 + nickels * 5);

        int pennies = (int) (change + .5);
        if (pennies > 0) {
            System.out.println(pennies + " PENNY(S)");
        }

    }

    private void intro() {
        System.out.println(simulateTabs(33) + "CHANGE");
        System.out.println(simulateTabs(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
        System.out.println();
        System.out.println("I, YOUR FRIENDLY MICROCOMPUTER, WILL DETERMINE");
        System.out.println("THE CORRECT CHANGE FOR ITEMS COSTING UP TO $100.");
        System.out.println();
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     * Converts input to a Double
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private double displayTextAndGetNumber(String text) {
        return Double.parseDouble(displayTextAndGetInput(text));
    }

    /*
     * Print a message on the screen, then accept input from Keyboard.
     *
     * @param text message to be displayed on screen.
     * @return what was typed by the player.
     */
    private String displayTextAndGetInput(String text) {
        System.out.print(text);
        return kbScanner.next();
    }

    /**
     * Simulate the old basic tab(xx) command which indented text by xx spaces.
     *
     * @param spaces number of spaces required
     * @return String with number of spaces
     */
    private String simulateTabs(int spaces) {
        char[] spacesTemp = new char[spaces];
        Arrays.fill(spacesTemp, ' ');
        return new String(spacesTemp);
    }
}

```

# `22_Change/java/src/ChangeGame.java`



这段代码定义了一个名为 `ChangeGame` 的类，其中包含一个名为 `main` 的方法。

在 `main` 方法中，使用 `new` 关键字创建了一个名为 `Change` 的类对象，并将其赋值给变量 `change`。

接着，调用 `change.play()` 方法，这个方法可能是来实现游戏的玩法或者规则转换等操作。但由于没有提供具体的实现，无法得知 `change.play()` 方法具体做了什么。


```
public class ChangeGame {
    public static void main(String[] args) {
        Change change = new Change();
        change.play();
    }
}

```

# `22_Change/javascript/change.js`

这段代码定义了两个函数，分别是 `print` 和 `input`。

1. `print` 函数的作用是将一个字符串 `str` 打印到网页的输出窗口（document.getElementById("output")）中。具体实现是通过在文档中创建一个文本节点，然后将 `str` 字符串插入到该节点中。

2. `input` 函数的作用是从用户接收输入字符串。该函数会等待用户输入字符，然后将其存储在变量 `input_str` 中。函数使用 `Promise` 对象，并且在函数内部添加了一个事件监听器，以便在用户按回车键时接收输入。当用户按回车键时，函数会将 `input_str` 打印到页面上，并在打印后将其从文档中删除。然后，函数会继续等待用户输入更多字符。

注意：这段代码将输出窗口（document.getElementById("output")) 保留为只读格式，这意味着用户无法编辑该元素。


```
// CHANGE
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       input_element = document.createElement("INPUT");

                       print("? ");
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       document.getElementById("output").appendChild(input_element);
                       input_element.focus();
                       input_str = undefined;
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      document.getElementById("output").removeChild(input_element);
                                                      print(input_str);
                                                      print("\n");
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

```

It looks like the program is a simple cashier. The cashier takes in a customer's payment and an item's cost. It then calculates the change and updates the customer's payment accordingly. The program uses ASCII art to display the payment amount and the item's cost.



```
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// Main program
async function main()
{
    print(tab(33) + "CHANGE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("I, YOUR FRIENDLY MICROCOMPUTER, WILL DETERMINE\n");
    print("THE CORRECT CHANGE FOR ITEMS COSTING UP TO $100.\n");
    print("\n");
    print("\n");
    while (1) {
        print("COST OF ITEM");
        a = parseFloat(await input());
        print("AMOUNT OF PAYMENT");
        p = parseFloat(await input());
        c = p - a;
        m = c;
        if (c == 0) {
            print("CORRECT AMOUNT, THANK YOU.\n");
        } else {
            print("YOUR CHANGE, $" + c + "\n");
            d = Math.floor(c / 10);
            if (d)
                print(d + " TEN DOLLAR BILL(S)\n");
            c -= d * 10;
            e = Math.floor(c / 5);
            if (e)
                print(e + " FIVE DOLLAR BILL(S)\n");
            c -= e * 5;
            f = Math.floor(c);
            if (f)
                print(f + " ONE DOLLAR BILL(S)\n");
            c -= f;
            c *= 100;
            g = Math.floor(c / 50);
            if (g)
                print(g + " ONE HALF DOLLAR(S)\n");
            c -= g * 50;
            h = Math.floor(c / 25);
            if (h)
                print(h + " QUARTER(S)\n");
            c -= h * 25;
            i = Math.floor(c / 10);
            if (i)
                print(i + " DIME(S)\n");
            c -= i * 10;
            j = Math.floor(c / 5);
            if (j)
                print(j + " NICKEL(S)\n");
            c -= j * 5;
            k = Math.floor(c + 0.5);
            if (k)
                print(k + " PENNY(S)\n");
            print("THANK YOU, COME AGAIN.\n");
            print("\n");
            print("\n");
        }
    }
}

```

这是 C 语言中的一个程序，名为 "main"。程序的作用是运行程序代码并返回一个整数。

"main" 函数是 C 语言中的一个全局函数，它定义了程序的入口点。当程序运行时，首先会执行 "main" 函数中的代码，然后就绪程序将返回一个整数，这个整数就是程序的返回值。

通常情况下，程序的返回值对程序有着非常重要的作用，因为程序的返回值可以用来检查程序的运行结果。 "main" 函数也可以被赋值，这样就可以将特定的值返回给程序。


```
main();

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


# `22_Change/python/change.py`

这段代码是一个简单的 Python 函数，名为 `print_centered`，它接受一个字符串参数 `msg`，并将其打印为字符串中心向下的形式。具体实现包括以下几个步骤：

1. 定义一个名为 `PAGE_WIDTH` 的变量，值为 64，设置页面宽度为 64 像素。
2. 定义一个名为 `print_centered` 的函数，该函数接受一个字符串参数 `msg`。
3. 在函数内部，使用字符串切片算法计算出 `PAGE_WIDTH` 像素数除以 2 的商和余数，然后将它们作为参数传递给 `print` 函数，实现字符串向左和向右的扩展，使得字符串中心显示。
4. 在函数内部，使用 `spaces` 变量存储计算得到的字符串中的空格数量，然后将 `msg` 对象与其进行连接，并将连接后的字符串存储在 `spaces` 变量中。
5. 最后，在 `print` 函数内部，使用左对齐和居中对齐的方式，将 `spaces` 和 `msg` 连接起来并输出，使得字符串居中显示。


```
"""
CHANGE

Change calculator

Port by Dave LeCompte
"""

PAGE_WIDTH = 64


def print_centered(msg: str) -> None:
    spaces = " " * ((PAGE_WIDTH - len(msg)) // 2)
    print(spaces + msg)


```

这段代码定义了三个函数，以及一个名为 "print\_header" 的函数内功。这些函数的目的是创建一个可以打印标题和介绍内容的字符串。

1. print\_header 函数接受一个字符串参数 title，然后使用 print\_centered 函数将其打印为字符串中心。接着打印一个带主题的行，然后打印一行垂直居中的行和一个类似于 "GET并不是完全免费！" 的消息。

2. print\_introduction 函数接受一个空字符串，然后打印一行垂直居中的行，接着打印一行消息，然后第二行垂直居中的行和一个类似于 "I'm your friendly microcomputer, and I'll determine the correct change for items costing up to $100." 的消息。

3. pennies\_to\_dollar\_string 函数接受一个浮点数参数 p，然后将其转换为美元并返回。它将浮点数 d 转换为小数点后两位的浮点数，然后将其打印为 "{d:0.2f}" 的字符串。


```
def print_header(title: str) -> None:
    print_centered(title)
    print_centered("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")


def print_introduction() -> None:
    print("I, YOUR FRIENDLY MICROCOMPUTER, WILL DETERMINE")
    print("THE CORRECT CHANGE FOR ITEMS COSTING UP TO $100.\n\n")


def pennies_to_dollar_string(p: float) -> str:
    d = p / 100
    return f"${d:0.2f}"


```

This is a Python function that calculates the change in pennies when a certain amount of money is shortened. The shorted amount is given as a decimal number with two significant figures. The function then converts the decimal number to a string with two commas and then uses string formatting to print the change in pennies with a specific number of decimal places.

The function takes two arguments: the shorted amount as a decimal number, and the number of decimal places to which the change in pennies should be printed. The function uses a while loop to iterate through all possible combinations of shorted amount and number of decimal places, and then短ens the decimal number to the appropriate number of decimal places before printing the change.

If the shorted amount is negative, the function prints a message and returns. If the shorted amount is zero, the function prints a message and returns.


```
def compute_change() -> None:
    print("COST OF ITEM?")
    cost = float(input())
    print("AMOUNT OF PAYMENT?")
    payment = float(input())

    change_in_pennies = round((payment - cost) * 100)
    if change_in_pennies == 0:
        print("CORRECT AMOUNT, THANK YOU.")
        return

    if change_in_pennies < 0:
        short = -change_in_pennies / 100

        print(f"SORRY, YOU HAVE SHORT-CHANGED ME ${short:0.2f}")
        print()
        return

    print(f"YOUR CHANGE, {pennies_to_dollar_string(change_in_pennies)}")

    d = change_in_pennies // 1000
    if d > 0:
        print(f"{d} TEN DOLLAR BILL(S)")
    change_in_pennies -= d * 1000

    e = change_in_pennies // 500
    if e > 0:
        print(f"{e} FIVE DOLLAR BILL(S)")
    change_in_pennies -= e * 500

    f = change_in_pennies // 100
    if f > 0:
        print(f"{f} ONE DOLLAR BILL(S)")
    change_in_pennies -= f * 100

    g = change_in_pennies // 50
    if g > 0:
        print("ONE HALF DOLLAR")
    change_in_pennies -= g * 50

    h = change_in_pennies // 25
    if h > 0:
        print(f"{h} QUARTER(S)")
    change_in_pennies -= h * 25

    i = change_in_pennies // 10
    if i > 0:
        print(f"{i} DIME(S)")
    change_in_pennies -= i * 10

    j = change_in_pennies // 5
    if j > 0:
        print(f"{j} NICKEL(S)")
    change_in_pennies -= j * 5

    if change_in_pennies > 0:
        print(f"{change_in_pennies} PENNY(S)")


```

这段代码是一个Python程序，名为"main"。程序的主要作用是输出"CHANGE"并介绍电影的背景信息，然后进入一个无限循环，在循环中调用"compute_change"函数来计算电影评分。

具体来说，这段代码首先定义了一个名为"main"的函数，该函数包含一个空括号。"print_header"函数用于输出一个"CHANGE"字符，然后调用"print_introduction"函数来介绍电影的背景信息。这两个函数的功能没有在代码中直接实现，但我们可以猜测它们会在程序中扮演重要角色。

接下来，程序进入一个无限循环，该循环将无限重复执行"compute_change"函数并输出"THANK YOU, COME AGAIN。"字符。每次调用"compute_change"函数时，程序会计算一部电影的评分，并输出一个感恩的话语。

最后，程序使用"if __name__ == "__main__":"这个条件来确保当程序直接运行时，会调用程序中的函数，否则不会执行任何函数。如果程序被传送到内存中作为独立模块，那么程序将不会调用任何函数，而是直接执行"print_header"和"print_introduction"函数。


```
def main() -> None:
    print_header("CHANGE")
    print_introduction()

    while True:
        compute_change()
        print("THANK YOU, COME AGAIN.\n\n")


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Anthony Rubick [AnthonyMichaelTDM](https://github.com/AnthonyMichaelTDM)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Checkers

This program plays checkers. The pieces played by the computer are marked with an “X”, yours are marked “O”. A move is made by specifying the coordinates of the piece to be moved (X, Y). Home (0,0) is in the bottom left and X specifies distance to the right of home (i.e., column) and Y specifies distance above home (i.e. row). You then specify where you wish to move to.

The original version of the program by Alan Segal was not able to recognize (or permit) a double or triple jump. If you tried one, it was likely that your piece would disappear altogether!

Steve North of Creative Computing rectified this problem and Lawrence Neal contributed modifications to allow the program to tell which player has won the game. The computer does not play a particularly good game but we leave it to _you_ to improve that.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=40)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=55)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

The file `checkers.annotated.bas` contains an indented and annotated
version of the source code.  This is no longer valid BASIC code but
should be more readable.

#### Porting Notes

 - If the computer moves a checker to the bottom row, it promotes, but
   leaves the original checker in place. (See line 1240)
 - Human players may move non-kings as if they were kings. (See lines 1590 to 1810)
 - Human players are not required to jump if it is possible, and may make a number
   other illegal moves (jumping their own pieces, jumping empty squares, etc.).
 - Curious writing to "I" variable without ever reading it. (See lines 1700 and 1806)


# `23_Checkers/csharp/Program.cs`

这段代码是一个用 C# 编写的检查ers程序，它用于在 Atari 游戏引擎中检查游戏逻辑。程序的主要目的是将 BASIC 游戏中的逻辑转换为 C# 代码，以便更容易理解和维护。

程序中没有使用任何面向对象的设计模式，而是使用仅有的函数 top-level 编写代码。这样可以更容易地学习 C# 语法，而不必了解面向对象的编程概念。

程序中包含了一些基本数据结构，如 multidimensional arrays，tuples 和 nullables，以及循环（for，foreach，while 和 do）。还使用了 C# 中的一些概念，如 IEnumerable。

该程序的作用是将 BASIC 游戏中的逻辑转换为 C# 代码，以方便在 Atari 游戏引擎中维护和修改。


```
﻿/*********************************************************************************
 * CHECKERS
 * ported from BASIC https://www.atariarchives.org/basicgames/showpage.php?page=41
 *
 * Porting philosophy
 * 1) Adhere to the original as much as possible
 * 2) Attempt to be understandable by Novice progammers
 *
 * There are no classes or Object Oriented design patterns used in this implementation.
 * Everything is written procedurally, using only top-level functions. Hopefully, this
 * will be approachable for someone who wants to learn C# syntax without experience with
 * Object Oriented concepts. Similarly, basic data structures have been chosen over more
 * powerful collection types.  Linq/lambda syntax is also excluded.
 *
 * C# Concepts contained in this example:
 *    Loops (for, foreach, while, and do)
 *    Multidimensional arrays
 *    Tuples
 *    Nullables
 *    IEnumerable (yield return / yield break)
 *
 * The original had multiple implementations of logic, like determining valid jump locations.
 * This has been refactored to reduce unnecessary code duplication.
 *********************************************************************************/
```

这段代码定义了两个函数，一个是 `SkipLines`，另一个是 `PrintBoard`。这两个函数的功能分别如下：

```java
// Display functions
void SkipLines(int count)
{
   for (int i = 0; i < count; i++)
   {
       Console.WriteLine();
   }
}

// Print board function
void PrintBoard(int state[][8])
{
   SkipLines(3);
   for (int y = 7; y >= 0; y--)
   {
       for (int x = 0; x < 8; x++)
       {
           switch(state[x,y])
           {
               case -2:
                   Console.Write("X*");
                   break;
               case -1:
                   Console.Write("X ");
                   break;
               case 0:
                   Console.Write(". ");
                   break;
               case 1:
                   Console.Write("O ");
                   break;
               case 2:
                   Console.Write("O*");
                   break;
           }
           Console.Write("   ");
       }
       Console.WriteLine();
   }
}
```

首先，这两个函数都是 `void` 类型的函数，即它们都可以作为函数体接受任意数量参数。

这两个函数的实现主要涉及到如何在控制台上输出文本。具体来说，第一个函数 `SkipLines` 通过循环来打印指定的行数，然后每行循环结束后，在控制台输出一个换行符。第二个函数 `PrintBoard` 通过循环来打印指定的二维矩阵中的每个元素，然后每行循环结束后，在控制台输出一个换行符。

在 `PrintBoard` 函数中，使用了一个 `switch` 语句来根据 `state` 数组中的每个元素的值，输出相应的字符。`state` 数组是一个二维数组，对于每个元素，需要根据它的值在 `switch` 中进行判断，并输出对应的字符。如果是 `-2`，则输出一个星号 `*`，如果是 `-1`，则输出一个 `X` 符号，如果是 `0`，则输出一个点 `.`，如果是 `1` 或 `2`，则输出一个 `O` 符号。


```
#region Display functions
void SkipLines(int count)
{
    for (int i = 0; i < count; i++)
    {
        Console.WriteLine();
    }
}

void PrintBoard(int[,] state)
{
    SkipLines(3);
    for (int y = 7; y >= 0; y--)
    {
        for (int x = 0; x < 8; x++)
        {
            switch(state[x,y])
            {
                case -2:
                    Console.Write("X*");
                    break;
                case -1:
                    Console.Write("X ");
                    break;
                case 0:
                    Console.Write(". ");
                    break;
                case 1:
                    Console.Write("O ");
                    break;
                case 2:
                    Console.Write("O*");
                    break;
            }
            Console.Write("   ");
        }
        Console.WriteLine();
    }
}

```

这段代码定义了三个函数，其中两个是 `void` 函数，另一个是 `ComputerWins()` 和 `PlayerWins()`，它们都是 `void` 函数。

`void WriteCenter()` 函数接受一个字符串参数 `text`，并计算出在这种文字中，如何将这个字符串划分成行，每行长度为 `LineLength`，然后使用这个长度减去 `text` 的长度，来计算出需要多少空格来在行之间插入 `spaces` 行。最后，使用 `Console.WriteLine()` 函数来输出结果，这个结果是在 `spaces` 和 `text` 之间插入行分隔符，并在 `spaces` 行首加上一个空格。

`ComputerWins()` 和 `PlayerWins()` 函数都是 `void` 函数，但是它们的功能是相反的。 `ComputerWins()` 函数使用 `Console.WriteLine()` 函数输出 "I WIN."，而 `PlayerWins()` 函数使用 `Console.WriteLine()` 函数输出 "YOU WIN."。

总的来说，这些函数的主要目的是在控制台输出一些文字，并在必要时通过 `spaces` 行来分隔不同的行。


```
void WriteCenter(string text)
{
    const int LineLength = 80;
    var spaces = (LineLength - text.Length) / 2;
    Console.WriteLine($"{"".PadLeft(spaces)}{text}");
}

void ComputerWins()
{
    Console.WriteLine("I WIN.");
}
void PlayerWins()
{
    Console.WriteLine("YOU WIN.");
}

```



这段代码是一个名为 "WriteIntroduction" 的函数，用于在游戏开始时向屏幕输出一些信息。具体来说，它输出的内容如下：

"THIS IS THE GAME OF CHECKERS. THE COMPPUTER IS X,"

"AND YOU ARE O. THE COMPUTER WILL MOVE FIRST."

"SQUARES ARE REFERRED TO BY A COORDINATE SYSTEM."

"(0,0) IS THE LOWER LEFT CORNER,"

"(0,7) IS THE UPPER LEFT CORNER,"

"(7,0) IS THE LOWER RIGHT CORNER,"

"(7,7) IS THE UPPER RIGHT CORNER,"

"THE COMPUTER WILL TYPE '+TO' WHEN YOU HAVE ANOTHER,"

"JUMP. TYPE TWO NEGATIVE NUMBERS IF YOU CANNOT JUMP."

"THIS IS THE GAME OF CHECKERS. THE COMPUTER WILL MOVE FIRST."

"BY YOURford立冬。"(这是游戏的简介，不是代码的一部分，但可能会在某些游戏中以这种方式出现)

"THIS IS THE GAME OF CHECKERS. THE COMPUTER WILL MOVE FIRST。"

"AND YOU ARE O. THE COMPUTER WILL MOVE FIRST."

"SQUARES ARE REFERRED TO BY A COORDINATE SYSTEM."

"(0,0) IS THE LOWER LEFT CORNER,"

"(0,7) IS THE UPPER LEFT CORNER,"

"(7,0) IS THE LOWER RIGHT CORNER,"

"(7,7) IS THE UPPER RIGHT CORNER,"

"THE COMPUTER WILL TYPE '+TO' WHEN YOU HAVE ANOTHER,"

"JUMP. TYPE TWO NEGATIVE NUMBERS IF YOU CANNOT JUMP."

"THIS IS THE GAME OF CHECKERS. THE COMPUTER WILL MOVE FIRST."

"THIS IS THE GAME OF CHECKERS. THE COMPUTER WILL MOVE FIRST."

"THIS IS THE GAME OF CHECKERS. THE COMPUTER WILL MOVE FIRST."

"THIS IS THE GAME OF CHECKERS. THE COMPUTER WILL MOVE FIRST."

"THIS IS THE GAME OF CHECKERS. THE COMPUTER WILL MOVE FIRST."

"THIS IS THE GAME OF CHECKERS. THE COMPUTER WILL MOVE FIRST."


```
void WriteIntroduction()
{
    WriteCenter("CHECKERS");
    WriteCenter("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    SkipLines(3);
    Console.WriteLine("THIS IS THE GAME OF CHECKERS. THE COMPUTER IS X,");
    Console.WriteLine("AND YOU ARE O.  THE COMPUTER WILL MOVE FIRST.");
    Console.WriteLine("SQUARES ARE REFERRED TO BY A COORDINATE SYSTEM.");
    Console.WriteLine("(0,0) IS THE LOWER LEFT CORNER");
    Console.WriteLine("(0,7) IS THE UPPER LEFT CORNER");
    Console.WriteLine("(7,0) IS THE LOWER RIGHT CORNER");
    Console.WriteLine("(7,7) IS THE UPPER RIGHT CORNER");
    Console.WriteLine("THE COMPUTER WILL TYPE '+TO' WHEN YOU HAVE ANOTHER");
    Console.WriteLine("JUMP.  TYPE TWO NEGATIVE NUMBERS IF YOU CANNOT JUMP.");
    SkipLines(3);
}
```

这段代码定义了三个状态验证函数，用于检查输入数据是否符合规则。

第一个函数 IsPointOutOfBounds(int x) 用于检查 x 是否在数轴上不在其预期的位置。

第二个函数 IsOutOfBounds((int x, int y) position) 在IsPointOutOfBounds()函数的基础上，添加了一个判断 y 是否在规定范围内的语句，以确定该位置是否超出限制。

第三个函数 IsJumpMove((int x, int y) from, (int x, int y) to) 用于检查从一个位置跳跃到另一个位置是否可以实现无缝的垂直移动。

注意：虽然这些函数的名称暗示了它们的功能，但它们实际上检查的是输入数据是否符合规定的范围，而不是在实现某种功能。


```
#endregion

#region State validation functions
bool IsPointOutOfBounds(int x)
{
    return x < 0 || x > 7;
}

bool IsOutOfBounds((int x, int y) position)
{
    return IsPointOutOfBounds(position.x) || IsPointOutOfBounds(position.y);
}

bool IsJumpMove((int x, int y) from, (int x, int y) to)
{
    return Math.Abs(from.y - to.y) == 2;
}

```

这段代码是一个条件判断函数，用于检查给定的玩家移动是否合法。函数接收一个二维状态数组，以及一个起始点和一个目标点，并返回一个布尔值。

具体来说，函数首先检查起始点和目标点是否为空，如果不是，则表示移动是合法的。接着，函数计算从起始点到目标点的垂直方向变化量（即 deltaY），并检查这个变化量是否等于 1 或 2。如果是，表示玩家只能向下或向左移动，因此函数返回 false。如果不是，则表示玩家可以向上或向右移动，函数继续判断。

接下来，函数再检查玩家是否可以向下移动，如果起始点是国王（用数字 1 表示），并且目标点不是空，则表示玩家只能向下移动，因此函数返回 false。如果目标点也是国王，则函数返回 true。

最后，函数还检查玩家是否可以跳跃。跳跃需要玩家有可跳的棋子（用数字 0 表示），并且起始点不是国王。如果起始点是国王并且目标点是空，或者玩家没有可跳的棋子，则函数返回 false。如果起始点不是国王，并且 deltaX 是 2，则函数返回 true，否则返回 false。


```
bool IsValidPlayerMove(int[,] state, (int x, int y) from, (int x, int y) to)
{
    if (state[to.x, to.y] != 0)
    {
        return false;
    }
    var deltaX = Math.Abs(to.x - from.x);
    var deltaY = Math.Abs(to.y - from.y);
    if (deltaX != 1 && deltaX != 2)
    {
        return false;
    }
    if (deltaX != deltaY)
    {
        return false;
    }
    if (state[from.x, from.y] == 1 && Math.Sign(to.y - from.y) <= 0)
    {
        // only kings can move downwards
        return false;
    }
    if (deltaX == 2)
    {
        var jump = GetJumpedPiece(from, to);
        if (state[jump.x, jump.y] >= 0)
        {
            // no valid piece to jump
            return false;
        }
    }
    return true;
}

```



这两段代码是用于判断游戏是否由玩家或计算机胜利的判断条件。

CheckForComputerWin函数接受一个二维整数State，表示游戏的当前状态。函数内部使用一个布尔变量playerAlive来记录玩家是否存活，每个状态元素都大于0时，playerAlive变量被设置为真，如果有一个大于0的元素，这个判断条件就成立了。

CheckForPlayerWin函数与CheckForComputerWin函数类似，但是要判断的是计算机是否存活，而不是游戏是否由计算机胜利。函数接受一个二维整数State，每个状态元素都小于0时，computerAlive变量被设置为真，如果有一个小于0的元素，这个判断条件就成立了。

两段代码的主要目的是判断游戏是否由玩家或计算机胜利，基于游戏当前的状态。


```
bool CheckForComputerWin(int[,] state)
{
    bool playerAlive = false;
    foreach (var piece in state)
    {
        if (piece > 0)
        {
            playerAlive = true;
            break;
        }
    }
    return !playerAlive;
}

bool CheckForPlayerWin(int[,] state)
{
    bool computerAlive = false;
    foreach (var piece in state)
    {
        if (piece < 0)
        {
            computerAlive = true;
            break;
        }
    }
    return !computerAlive;
}
```

这段代码是一个C#类，名为Board。这个类定义了一个名为“arithmetic”的矩形（即棋盘）数据结构，该数据结构包含了一些方法来获取和应用棋盘上的位置。

具体来说，这个类包含两个方法：

1. GetJumpedPiece方法，该方法接受两个位置参数“from”和“to”，并返回跳棋子所处的位置。这个方法实现了一个简单的算法，通过计算从原位置（from）到目标位置（to）的中点坐标，来找到跳棋子所处的位置。

2. ApplyDirectionalVector方法，该方法接受一个名为“direction”的方向向量和一个位置参数“from”。该方法将方向向量与从位置（from）应用，并返回结果位置。这个方法可以用于移动棋子，向指定方向移动棋子指定数量单位。


```
#endregion

#region Board "arithmetic"
/// <summary>
/// Get the Coordinates of a jumped piece
/// </summary>
(int x, int y) GetJumpedPiece((int x, int y) from, (int x, int y) to)
{
    var midX = (to.x + from.x) / 2;
    var midY = (to.y + from.y) / 2;
    return (midX, midY);
}
/// <summary>
/// Apply a directional vector "direction" to location "from"
/// return resulting location
```

这段代码定义了一个名为 GetLocation 的函数，它接收两个整数参数 x 和 y，以及一个方向（从下到上，从左到右）和一个位置参数 from 和 to。函数返回一个元组表示从位置 to 移动石头后的位置。

该函数将石头从位置 from 移动到位置 to，并将石头从位置 from 移除。如果移动是向上或向下的，则函数会相应地改变石头的方向。如果移动是跳跃，函数会尝试从位置 from 检索跳跃过的 piece，并从位置 to 删除该 piece。

函数 ApplyMove 用于更新游戏的地图状态。它接收一个表示当前状态的二维数组 state，以及一个起始位置 from 和目标位置 to。函数会遍历 state 数组中的每个元素，并将从位置 to 移动石头的位置存储在 state[to.x, to.y] 中。然后，它遍历 state 数组中的每个元素，并将 from.x 和 from.y 设置为 0。如果移动是跳跃，函数会从 state 数组中移除跳跃过的 piece，并更新跳跃位置 state[jump.x, jump.y] 为 0。

总之，这段代码定义了一个用于更新游戏地图的函数，用于在给定位置进行向上、向下、跳跃等操作。


```
/// direction will contain: (-1,-1), (-1, 1), ( 1,-1), ( 1, 1)
/// /// </summary>
(int x, int y) GetLocation((int x , int y) from, (int x, int y) direction)
{
    return (x: from.x + direction.x, y: from.y + direction.y);
}
#endregion

#region State change functions
/// <summary>
/// Alter current "state" by moving a piece from "from" to "to"
/// This method does not verify that the move being made is valid
/// This method works for both player moves and computer moves
/// </summary>
int[,] ApplyMove(int[,] state, (int x, int y) from, (int x, int y) to)
{
    state[to.x, to.y] = state[from.x, from.y];
    state[from.x, from.y] = 0;

    if (IsJumpMove(from, to))
    {
        // a jump was made
        // remove the jumped piece from the board
        var jump = GetJumpedPiece(from, to);
        state[jump.x, jump.y] = 0;
    }
    return state;
}
```

这段代码的作用是检查游戏棋盘在什么时候玩家或电脑达到了“王”的位置，如果王的位置已经被双方占据了，则改变棋盘上的王的位置。具体来说，代码首先检查电脑是否占据了第 8 行，如果是，则将该行及其之后的 8 个元素设置为 -2；如果是玩家，则将该行及其之后的 8 个元素设置为 2。然后，代码检查每一行的最后一个位置，如果是电脑则不变，如果是玩家则变化为 2。这样，在游戏结束后，当玩家或电脑达到了“王”的位置，就可以通过改变该位置的元素来通知其它玩家或电脑。


```
/// <summary>
/// At the end of a turn (either player or computer) check to see if any pieces
/// reached the final row.  If so, change them to kings (crown)
/// </summary>
int[,] CrownKingPieces(int[,] state)
{
    for (int x = 0; x < 8; x++)
    {
        // check the bottom row if computer has a piece in it
        if (state[x, 0] == -1)
        {
            state[x, 0] = -2;
        }
        // check the top row if the player has a piece in it
        if (state[x, 7] == 1)
        {
            state[x, 7] = 2;
        }
    }
    return state;
}
```

这段代码定义了一个名为`GetCandidateMove`的函数，它接受一个状态矩阵`state`，以及一个起始位置`from`和一个方向`direction`作为参数。它的作用是判断在给定的向径`direction`下，起始位置`from`是否可以进行移动，并在需要时返回移动后的位置。

函数的具体实现可以分为以下几个步骤：

1. 从状态矩阵`state`中得到起始位置`from`和一个方向`direction`所处的位置，即`to`。
2. 判断`to`是否在边界位置（行列坐标轴的左边界、下边界、上边界或右边界）。如果是，返回`null`。
3. 如果`to`不在边界位置，首先检查`direction`是否与`state`中的某个位置匹配。如果不匹配，继续判断`direction`是否可以沿着`state`中的位置移动。如果可以，进行移动并再次判断`to`是否在边界位置。如果仍然不在边界位置，说明有空间可以移动，返回移动后的位置。
4. 如果`direction`可以移动且`to`不在边界位置，说明可以进行移动，返回移动后的位置。

总之，该函数的作用是判断在给定向径下，给定起始位置是否可以进行移动，并在需要时返回移动后的位置。


```
#endregion

#region Computer Logic
/// <summary>
/// Given a current location "from", determine if a move exists in a given vector, "direction"
/// direction will contain: (-1,-1), (-1, 1), ( 1,-1), ( 1, 1)
/// return "null" if no move is possible in this direction
/// </summary>
(int x, int y)? GetCandidateMove(int[,] state, (int x, int y) from, (int x, int y) direction)
{
    var to = GetLocation(from, direction);
    if (IsOutOfBounds(to))
        return null;
    if (state[to.x, to.y] > 0)
    {
        // potential jump
        to = GetLocation(to, direction);
        if (IsOutOfBounds(to))
            return null;
    }
    if (state[to.x, to.y] != 0)
        // space already occupied by another piece
        return null;

    return to;
}
```

这段代码定义了一个名为RankMove的函数，其作用是计算一个棋盘状态下的 Move 棋的排名。该函数接收一个棋盘状态数组 state，以及一个从 (x, y) 到 (x, y) 的坐标范围 from 和 to，并返回一个整数表示该棋的排名。

以下是函数的实现细节：

1. 函数开始时，将 rank 初始化为 0。
2. 如果 to 坐标为空，并且状态数组中对应位置的值等于 -1，则认为这是一个国王棋，将 rank 加 2。
3. 如果IsJumpMove函数返回 true，说明这是一个跳跃（即 jump），将 rank 加 5。
4. 如果 from 坐标为 7，则认为这个棋是往右跨了一格，将 rank 减 2。
5. 如果 to 坐标为 0 或 7，则认为这个棋是往左或上跨了一格，将 rank 加 1。
6. 遍历从 2 到 8（不包括 8）的所有位置，计算在目标位置前方的所有位置，如果这些位置中有 -1，说明被保护了，rank 加 1。如果所有位置均非负，则 rank 加 1。
7. 返回 rank。

该函数的作用是计算一个棋盘状态下的一个跳跃棋的排名，跳跃棋可以连接到其他棋，但无法跨越棋盘跨越到空位置。


```
/// <summary>
/// Calculate a rank for a given potential move
/// The higher the rank value, the better the move is considered to be
/// </summary>
int RankMove(int[,] state, (int x, int y) from, (int x, int y) to)
{
    int rank = 0;

    if (to.y == 0 && state[from.x, from.y] == -1)
    {
        // getting a king
        rank += 2;
    }
    if (IsJumpMove(from, to))
    {
        // making a jump
        rank += 5;
    }
    if (from.y == 7)
    {
        // leaving home row
        rank -= 2;
    }
    if (to.x == 0 || to.x == 7)
    {
        // move to edge of board
        rank += 1;
    }
    // look to the row in front of the potential destination for
    for (int c = -1; c <=1; c+=2)
    {
        var inFront = GetLocation(to, (c, -1));
        if (IsOutOfBounds(inFront))
            continue;
        if (state[inFront.x, inFront.y] < 0)
        {
            // protected by our piece in front
            rank++;
            continue;
        }
        var inBack = GetLocation(to, (-c, 1));
        if (IsOutOfBounds(inBack))
        {
            continue;
        }
        if ((state[inFront.x, inFront.y] > 0) &&
            (state[inBack.x, inBack.y] == 0) || (inBack == from))
        {
            // the player can jump us
            rank -= 2;
        }
    }
    return rank;
};

```

这段代码定义了一个名为 `GetPossibleMoves` 的函数，它接受一个 2D 状态表示，以及一个起始位置 `(x, y)`，并返回一个可能的移动列表。

该函数首先定义了一个枚举类型 `IEnumerable<(int x, int y)>`，用于表示可能的移动。然后，函数检查起始位置 `(x, y)` 是否属于其自身，如果是，则函数将返回一个只包含 `null` 的集合，表示没有可能的移动。否则，函数将返回一个包含符合条件移动的集合。

接下来，函数使用两个循环来遍历可能的移动。第一个循环确定移动的方向（根据 `state[from.x, from.y]` 的值，如果 `-1` 出现在数组中，就表示向相反方向移动），第二个循环在移动方向确定后，再次检查是否有一种可行的移动。如果找到了一种可行的移动，函数将其添加到移动列表中。

最后，函数使用 `yield` 语句将移动列表中的每个元素 yields，这样可以在需要时异步地获取移动列表中的元素。


```
/// <summary>
/// Returns an enumeration of possible moves that can be made by the given piece "from"
/// If no moves, can be made, the enumeration will be empty
/// </summary>
IEnumerable<(int x, int y)> GetPossibleMoves(int[,] state, (int x, int y) from)
{
    int maxB;
    switch (state[from.x, from.y])
    {
        case -2:
            // kings can go backwards too
            maxB = 1;
            break;
        case -1:
            maxB = -1;
            break;
        default:
            // not one of our pieces
            yield break;
    }

    for (int a = -1; a <= 1; a += 2)
    {
        // a
        // -1 = left
        // +1 = right
        for (int b = -1; b <= maxB; b += 2)
        {
            // b
            // -1 = forwards
            // +1 = backwards (only kings allowed to make this move)
            var to = GetCandidateMove(state, from, (a, b));
            if (to == null)
            {
                // no valid move in this direction
                continue;
            }
            yield return to.Value;
        }
    }
}
```

该代码的作用是确定一个棋盘状态（state）下的最佳移动（bestMove），即使在这种情况下可能的移动（possibleMoves）列表中没有任何有效的移动，返回null。

具体来说，该代码使用以下步骤来实现：

1. 初始化两个变量bestRank和bestMove为null。
2. 遍历可用的移动（possibleMoves）列表。
3. 对于每个移动（move），计算在当前状态下进行该移动后的新排名（rank）。
4. 如果当前状态下的最佳排名（bestRank）为null，或者新排名（rank）大于最佳排名（bestRank），那么将最佳排名（bestRank）设置为新排名（rank），并将移动（move）作为最佳移动（bestMove）。
5. 返回最佳移动（bestMove）。

这里需要注意的是，该代码使用了C#的null类型。在实际应用中，应该根据具体的需求来选择是否使用null类型。


```
/// <summary>
/// Determine the best move from a list of candidate moves "possibleMoves"
/// Returns "null" if no move can be made
/// </summary>
((int x, int y) from, (int x, int y) to)? GetBestMove(int[,] state, IEnumerable<((int x, int y) from, (int x, int y) to)> possibleMoves)
{
    int? bestRank = null;
    ((int x, int y) from, (int x, int y) to)? bestMove = null;

    foreach (var move in possibleMoves)
    {
        int rank = RankMove(state, move.from, move.to);

        if (bestRank == null || rank > bestRank)
        {
            bestRank = rank;
            bestMove = move;
        }
    }

    return bestMove;
}

```

这段代码的主要目的是检查整个棋盘并找到所有可能的移动，如果找到了，则返回该移动，否则返回 "null"。它接受一个整数数组 `state` 作为输入，并使用以下两个函数来获取棋盘上的所有可能的移动：

1. `GetPossibleMoves(state, from)`：该函数使用递归方式遍历棋盘，并返回一个包含所有可能的移动的列表。它从棋盘的左上角开始，以 8 步为一步，递归地遍历所有可能的移动。
2. `GetBestMove(state, possibleMoves)`：该函数接收一个整数数组 `state` 和一个包含所有可能的移动的列表 `possibleMoves`。它使用以下算法找到所有可能的移动中最好的一个，并返回该移动：

1. 从 `possibleMoves` 列表中找到第一个匹配 `from` 的移动。
2. 如果找到了匹配的移动，则将其从 `possibleMoves` 列表中删除。
3. 否则，返回 `null`，表示没有找到任何匹配的移动。

最后，该函数将调用 `GetBestMove(state, possibleMoves)` 函数来查找所有可能的移动中最好的一个，并将其返回。


```
/// <summary>
/// Examine the entire board and record all possible moves
/// Return the best move found, if one exists
/// Returns "null" if no move found
/// </summary>
((int x, int y) from, (int x, int y) to)? CalculateMove(int[,] state)
{
    var possibleMoves = new List<((int x, int y) from, (int x, int y) to)>();
    for (int x = 0; x < 8; x++)
    {
        for (int y = 0; y < 8; y++)
        {
            var from = (x, y);
            foreach (var to in GetPossibleMoves(state, from))
            {
                possibleMoves.Add((from, to));
            }
        }
    }
    var bestMove = GetBestMove(state, possibleMoves);
    return bestMove;
}

```

这段代码定义了一个名为`ComputerTurn`的函数，用于模拟电脑在游戏中的决策过程。该函数接收一个由整数组成的棋盘状态`state`，并输出电脑是否可以进行移动以及可能的移动。

函数内部首先定义了一个变量`moveMade`，用于记录是否进行了移动，然后调用一个名为`CalculateMove`的函数来尝试找到一个可移动的位置。如果找到了移动，函数将`moveMade`设置为`true`，并从状态中移除该移动。

接下来，函数使用`from`变量来记录当前移动的位置，然后输出该位置以及移动类型（平移或翻转）。接着，函数检查输入的移动是否可以跳过其他棋子，如果是，则函数将其添加到`possibleMoves`列表中。然后，函数使用`GetBestMove`函数来查找`possibleMoves`列表中的最佳移动，并将结果设置给`move`变量。

接下来，函数使用`while`循环来不断尝试新的移动，并检查输入的移动是否可以跳过其他棋子。如果移动不被允许，函数退出循环。

最后，函数应用 Crown 王冠状态，并返回`true` 表示状态有效，以及一个包含状态的数组`state`。


```
/// <summary>
/// The logic behind the Computer's turn
/// Look for valid moves and possible subsequent moves
/// </summary>
(bool moveMade, int[,] state) ComputerTurn(int[,] state)
{
    // Get best move available
    var move = CalculateMove(state);
    if (move == null)
    {
        // No move can be made
        return (false, state);
    }
    var from = move.Value.from;
    Console.Write($"FROM {from.x} {from.y} ");
    // Continue to make moves until no more valid moves can be made
    while (move != null)
    {
        var to = move.Value.to;
        Console.WriteLine($"TO {to.x} {to.y}");
        state = ApplyMove(state, from, to);
        if (!IsJumpMove(from, to))
            break;

        // check for double / triple / etc. jump
        var possibleMoves = new List<((int x, int y) from, (int x, int y) to)>();
        from = to;
        foreach (var candidate in GetPossibleMoves(state, from))
        {
            if (IsJumpMove(from, candidate))
            {
                possibleMoves.Add((from, candidate));
            }
        }
        // Get best jump move
        move = GetBestMove(state, possibleMoves);
    }
    // apply crowns to any new Kings
    state = CrownKingPieces(state);
    return (true, state);
}
```

这段代码是一个C#类，名为“Player Logic”。该类包含一个名为“GetCoordinate”的函数，用于获取玩家输入的位置坐标。函数接受一个字符串参数“提示”，可以根据这个提示判断输入是否正确，如果输入正确，则返回包含位置坐标的元组，否则返回null。

具体来说，该函数首先通过从字符串中分离出两个整数部分来获取输入的x和y坐标，接着使用int.TryParse方法尝试将输入的字符串转换为整数。如果转换成功，函数将返回包含坐标位置的元组，否则返回null。


```
#endregion

#region Player Logic
/// <summary>
/// Get input from the player in the form "x,y" where x and y are integers
/// If invalid input is received, return null
/// If input is valid, return the coordinate of the location
/// </summary>
(int x, int y)? GetCoordinate(string prompt)
{
    Console.Write(prompt + "? ");
    var input = Console.ReadLine();
    // split the string into multiple parts
    var parts = input?.Split(",");
    if (parts?.Length != 2)
        // must be exactly 2 parts
        return null;
    int x;
    if (!int.TryParse(parts[0], out x))
        // first part is not a number
        return null;
    int y;
    if (!int.TryParse(parts[1], out y))
        //second part is not a number
        return null;

    return (x, y);
}

```

This is a function definition for a valid move in a chess game. It takes a state array that represents the current state of the game, including the pieces on the board and their ownership. The function is valid if the move is valid according to certain checks, such as ensuring that the piece being moved is owned by the player and that the FROM and TO squares are within a certain distance. If no valid move can be found, the function returns the original state. The function also includes some checks for the direction of the move, but these checks are not fully implemented in this version of the function.


```
/// <summary>
/// Get the move from the player.
/// return a tuple of "from" and "to" representing a valid move
///
/// </summary>
((int x, int y) from, (int x,int y) to) GetPlayerMove(int[,] state)
{
    // The original program has some issues regarding user input
    // 1)  There are minimal data sanity checks in the original:
    //     a)  FROM piece must be owned by player
    //     b)  TO location must be empty
    //     c)  the FROM and TO x's must be less than 2 squares away
    //     d)  the FROM and TO y's must be same distance as x's
    //     No checks are made for direction, if a jump is valid, or
    //     if the piece even moves.
    // 2)  Once a valid FROM is selected, a TO must be selected.
    //     If there are no valid TO locations, you are soft-locked
    // This approach is intentionally different from the original
    // but maintains the original intent as much as possible
    // 1)  Select a FROM location
    // 2)  If FROM is invalid, return to step 1
    // 3)  Select a TO location
    // 4)  If TO is invalid or the implied move is invalid,
    //     return to step 1


    // There is still currently no way for the player to indicate that no move can be made
    // This matches the original logic, but is a candidate for a refactor

    do
    {
        var from = GetCoordinate("FROM");
        if ((from != null)
            && !IsOutOfBounds(from.Value)
            && (state[from.Value.x, from.Value.y] > 0))
        {
            // we have a valid "from" location
            var to = GetCoordinate("TO");
            if ((to != null)
                && !IsOutOfBounds(to.Value)
                && IsValidPlayerMove(state, from.Value, to.Value))
            {
                // we have a valid "to" location
                return (from.Value, to.Value);
            }
        }
    } while (true);
}

```

这段代码的作用是获取玩家是否要进行一次后续跳跃。如果玩家输入负数作为跳跃的坐标，则该函数返回一个位置信息(from, to)。否则，如果玩家没有提供跳跃坐标或者坐标超出范围，则返回null。函数首先使用do-while循环来不断尝试玩家提供的跳跃坐标，每次尝试更新to坐标。如果to坐标不为空且玩家没有表明要停止跳跃，则函数将返回一个包含from和to坐标的元组。


```
/// <summary>
/// Get a subsequent jump from the player if they can / want to
/// returns a move ("from", "to") if a player jumps
/// returns null if a player does not make another move
/// The player must input negative numbers for the coordinates to indicate
/// that no more moves are to be made.  This matches the original implementation
/// </summary>
((int x, int y) from, (int x, int y) to)? GetPlayerSubsequentJump(int[,] state, (int x, int y) from)
{
    do
    {
        var to = GetCoordinate("+TO");
        if ((to != null)
            && !IsOutOfBounds(to.Value)
            && IsValidPlayerMove(state, from, to.Value)
            && IsJumpMove(from, to.Value))
        {
            // we have a valid "to" location
            return (from, to.Value); ;
        }

        if (to != null && to.Value.x < 0 && to.Value.y < 0)
        {
            // player has indicated to not make any more moves
            return null;
        }
    }
    while (true);
}

```

这段代码定义了一个名为 `PlayerTurn` 的函数，用于处理玩家在游戏中的回合。函数的参数是一个由 4 个整数组成的数组 `state`，表示游戏当前的状态。

函数的主要逻辑如下：

1. 从玩家获取输入，用于选择移动方向。
2. 如果可能，使用玩家输入获取后续跳跃，并将其应用到游戏状态中。
3. 如果玩家没有选择跳跃，那么游戏状态将不会发生变化，循环将继续进行。
4. 如果玩家选择了跳跃，那么将获取玩家后续跳跃的位置，如果尚未有跳跃，则创建一个新的跳跃。
5. 如果玩家没有进行跳跃，则在循环中继续进行，直到有玩家选择跳跃。
6. 在玩家选择跳跃后，使用获取的跳跃位置获取跳跃后的位置，并检查该位置是否为国王。
7. 最后，返回游戏的状态。

该函数可以被视为是游戏循环的一部分，用于处理玩家在游戏中的操作并返回新的游戏状态。


```
/// <summary>
/// The logic behind the Player's turn
/// Get the player input for a move
/// Get subsequent jumps, if possible
/// </summary>
int [,] PlayerTurn(int[,] state)
{
    var move = GetPlayerMove(state);
    do
    {
        state = ApplyMove(state, move.from, move.to);
        if (!IsJumpMove(move.from, move.to))
        {
            // If player doesn't make a jump move, no further moves are possible
            break;
        }
        var nextMove = GetPlayerSubsequentJump(state, move.to);
        if (nextMove == null)
        {
            // another jump is not made
            break;
        }
        move = nextMove.Value;
    }
    while (true);
    // check to see if any kings need crowning
    state = CrownKingPieces(state);
    return state;
}
```

这段代码是一个八行八列的二维数组，代表了游戏棋盘的状态。在初始化状态下，所有格子都是0，即没有任何生命体存在。

接下来，代码创建了一个包含8个元素，每个元素都是2的数组，代表电脑的棋子。再创建了一个包含8个元素，每个元素都是-1的数组，代表玩家的棋子。

为了更好地可视化游戏棋盘，代码将玩家的头部向右旋转了90度。

最后，代码将初始化状态下所有格子的值保存到了一个名为"state"的变量中。


```
#endregion

/*****************************************************************************
 *
 * Main program starts here
 *
 ****************************************************************************/

WriteIntroduction();

// initalize state -  empty spots initialize to 0
// set player pieces to 1, computer pieces to -1
// turn your head to the right to visualize the board.
// kings will be represented by -2 (for computer) and 2 (for player)
int[,] state = new int[8, 8] {
    { 1, 0, 1, 0, 0, 0,-1, 0 },
    { 0, 1, 0, 0, 0,-1, 0,-1 },
    { 1, 0, 1, 0, 0, 0,-1, 0 },
    { 0, 1, 0, 0, 0,-1, 0,-1 },
    { 1, 0, 1, 0, 0, 0,-1, 0 },
    { 0, 1, 0, 0, 0,-1, 0,-1 },
    { 1, 0, 1, 0, 0, 0,-1, 0 },
    { 0, 1, 0, 0, 0,-1, 0,-1 },
};

```

这段代码是一个无限循环，一直在玩一个简单的棋类游戏。在这个游戏中，玩家每轮会交替进行移动，而电脑则会在每一步中先移动。游戏的目标是让玩家或电脑中任意一方先赢得游戏。

具体来说，代码中实现了以下功能：

1. 初始化游戏状态，包括电脑和玩家的初始位置以及棋盘状态。

2. 通过调用 ComputerTurn() 函数来判断电脑是否可以获胜，即电脑是否可以移动并占据了优势位置。

3. 如果电脑无法获胜，则执行以下操作：

  a. 调用 ComputerWins() 函数来判断游戏是否平局。

  b. 输出当前棋盘状态。

  c. 如果电脑获胜，则执行以下操作：

    i. 调用 ComputerWins() 函数来判断游戏是否平局。

    ii. 调用 PlayerTurn() 函数来决定下一轮的移动方。

    iii. 如果玩家获胜，则执行以下操作：

      a. 调用 PlayerWins() 函数来判断游戏是否平局。

      b. 输出当前棋盘状态。

      c. 如果玩家获胜，则执行以下操作：

       i. 游戏平局，跳出循环。

       ii. 调用 ComputerTurn() 函数来决定下一轮的移动方。

       iii. 重复执行第一步的操作，继续进行游戏。

3. 如果电脑或玩家获胜，则跳出循环。

4. 初始化电脑的初始位置。

5. 初始化玩家的初始位置。


```
while (true)
{
    bool moveMade;
    (moveMade, state) = ComputerTurn(state);
    if (!moveMade)
    {
        // In the original program the computer wins if it cannot make a move
        // I believe the player should win in this case, assuming the player can make a move.
        // if neither player can make a move, the game should be draw.
        // I have left it as the original logic for now.
        ComputerWins();
        break;
    }
    PrintBoard(state);
    if (CheckForComputerWin(state))
    {
        ComputerWins();
        break;
    }
    state = PlayerTurn(state);
    if (CheckForPlayerWin(state))
    {
        PlayerWins();
        break;
    }
}

```