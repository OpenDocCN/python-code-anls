# BasicComputerGames源码解析 88

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)

I have replaced the manual date logic with Perl built-ins to the extent
possible. Unfortunately the kind of date math involved in the "time
spent doing ..." functionality is not well-defined, so I have been
forced to retain the original logic here. Sigh.

You can use any punctuation character you please in the date
input. So something like 2/29/2020 is perfectly acceptable.

It would also have been nice to produce a localized version that
supports day/month/year or year-month-day input, but that didn't happen.

Also nice would have been language-specific output -- especially if it
could have accommodated regional differences in which day of the week or
month is unlucky.

Tom Wyant


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


# `95_Weekday/python/weekday.py`

这段代码是一个函数，名为 `WEEKDAY`。它实现了以下功能：

1. 计算 entered 日期是星期几。
2. 估计进入生日的人会从事哪些活动，如果他们过了生日。
3. 计算退休年龄，假设在 65 岁退休。

代码中使用了 `datetime` 模块来处理日期和时间相关的计算。


```
"""
WEEKDAY

Calculates which weekday an entered date is.

Also estimates how long a person has done certain activities, if they
entered their birthday.

Also calculates the year of retirement, assuming retiring at age 65.

Ported by Dave LeCompte.
"""

import datetime
from typing import Tuple

```

该代码是一个Python程序，它的作用是获取用户输入的日期。程序会首先向用户提示输入日期，然后循环等待用户输入，直到用户输入正确为止。如果用户输入的日期格式不正确，程序会提示用户重新输入。程序会返回获取到的日期，格式为(月数， 日数， 年数)。

具体来说，代码可以分为以下几个部分：

1. `GET_TODAY_FROM_SYSTEM = True` 是一个Python设置，它用于设置一个布尔值，表示当前时间是否来自系统。

2. `def get_date_from_user(prompt: str) -> Tuple[int, int, int]:` 是一个函数，它接收一个字符串参数 prompt，用于提示用户输入日期。函数内部包含一个无限循环，用于不断向用户提示输入日期，直到用户输入正确为止。

3. `while True:` 是一个无限循环，它永远不会退出。

4. `print(prompt)` 在循环中，程序会打印 prompt，用于向用户显示输入提示。

5. `line = input("请输入日期（格式：YYYY-MM-DD）：')` 用于获取用户输入，它接收一个字符串类型的变量 line。

6. `try:` 用于尝试从 line 中解析出月份、日期和年份。

7. `month_num, day_num, year_num = (int(x) for x in line.split(","))` 将每个 x 值转换成整数，用于获取月份、日期和年份。

8. `return month_num, day_num, year_num` 返回月份、日期和年份。

9. `def get_date_from_system() -> Tuple[int, int, int]:` 另一个函数，它返回当前系统的日期。函数使用 datetime.datetime.today() 获取当前日期，并返回它的月份、日期和年份。

10. `dt = datetime.datetime.today()` 获取当前日期并将其存储在变量 dt 中。

11. `return dt.month, dt.day, dt.year` 返回获取到的当前日期，格式为(月数， 日数， 年数)。


```
GET_TODAY_FROM_SYSTEM = True


def get_date_from_user(prompt: str) -> Tuple[int, int, int]:
    while True:
        print(prompt)
        date_str = input()
        try:
            month_num, day_num, year_num = (int(x) for x in date_str.split(","))
            return month_num, day_num, year_num
        except Exception:
            print("I COULDN'T UNDERSTAND THAT. TRY AGAIN.")


def get_date_from_system() -> Tuple[int, int, int]:
    dt = datetime.datetime.today()
    return dt.month, dt.day, dt.year


```

这段代码定义了一个名为 `get_day_of_week` 的函数，它接受两个参数 `weekday_index` 和 `day`。函数的作用是返回 `day_names[weekday_index]`，其中 `weekday_index` 是从 0 到 7 的整数，表示一星期中的哪一天，`day` 是 0 到 7 的整数，表示一星期中的第几天。函数的实现主要通过查阅字典 `day_names` 来获取相应的字符串，具体实现是判断 `weekday_index` 是否为 6，如果是，并且 `day` 是否为 13，如果是，那么返回字符串 "FRIDAY THE THIRTEENTH---BEWARE!"。


```
def get_day_of_week(weekday_index, day) -> str:
    day_names = {
        1: "SUNDAY",
        2: "MONDAY",
        3: "TUESDAY",
        4: "WEDNESDAY",
        5: "THURSDAY",
        6: "FRIDAY",
        7: "SATURDAY",
    }

    if weekday_index == 6 and day == 13:
        return "FRIDAY THE THIRTEENTH---BEWARE!"
    return day_names[weekday_index]


```

这两段代码是在Python中定义的函数。

第一个函数 `previous_day` 的参数 `b` 代表一个整数，返回值为 `b` 减去 1。这个函数的作用是获取前一个日期，如果 `b` 为 0，则将 `b` 赋值为 6，即前一天的日期是 6 号。

第二个函数 `is_leap_year` 的参数 `year` 是一个整数，返回值为 `True` 或者 `False`。这个函数的作用是判断给定的年份是否为闰年，具体规则如下：

- 如果 `year` 能被 4 整除，则这个年份是闰年，返回 `True`。
- 如果 `year` 能被 100 整除，但不是 400 的倍数，则这个年份不是闰年，返回 `False`。
- 如果 `year` 既不能被 4 整除，也不能被 100 整除，则这个年份是平年，返回 `False`。

函数内部使用的是 `if` 语句，如果 `year % 4` 余数为 0，就执行第一个判断条件，否则执行第二个判断条件。如果 `year % 100` 余数为 0，但不是 400 的倍数，就执行第二个判断条件，否则执行第一个判断条件。以此类推，直到所有条件都不满足，返回 `False`。如果所有条件都满足，则返回 `True`。


```
def previous_day(b) -> int:
    if b == 0:
        b = 6
    return b - 1


def is_leap_year(year: int) -> bool:
    if (year % 4) != 0:
        return False
    if (year % 100) != 0:
        return True
    if (year % 400) != 0:
        return False
    return True


```

这段代码定义了三个函数，用于调整日期以适应闰年和指定月份。

第一个函数 `adjust_day_for_leap_year` 接受一个年份参数 `year`，并返回该年份的 `previous_day` 函数的输出。`previous_day` 函数接受一个日期参数 `b`，并返回该日期向前推的年份。如果这一年是闰年，函数将返回 `b` 减去 1，否则返回 `b`。函数首先检查传入的年份是否是闰年，如果是，则将 `b` 减去 1。

第二个函数 `adjust_weekday` 接受一个日期参数 `b`，以及一个月份参数 `month` 和一个年份参数 `year`。函数首先检查月份是否小于等于 2，如果是，则将 `b` 调整为对应年份的星期几。如果 `month` 不小于 2，函数将使用 `adjust_day_for_leap_year` 函数来调整日期。然后，函数将 `b` 赋值为 0（星期日）或 7（星期一），具体取决于 `month` 的值。

第三个函数 `calc_day_value` 接受一个年份参数 `year`、一个月份参数 `month` 和一个日期参数 `day`。函数计算指定年份的月份值，然后将当前年份乘以 31 再加上 `day`，最后将结果乘以 365 得到该日期的价值。


```
def adjust_day_for_leap_year(b, year):
    if is_leap_year(year):
        b = previous_day(b)
    return b


def adjust_weekday(b, month, year):
    if month <= 2:
        b = adjust_day_for_leap_year(b, year)
    if b == 0:
        b = 7
    return b


def calc_day_value(year, month, day):
    return (year * 12 + month) * 31 + day


```

这段代码是一个名为 `deduct_time` 的函数，它接受五个参数：`frac`（分数，表示为一个小数，可以向下取整）、`days`（天数，单位为整数，表示为 `days_remain` 的值）、`years_remain`（剩余的年数，单位为整数，表示为 `days_remain` 的值）、`months_remain`（剩余的月份，单位为整数，表示为 `days_remain` 的值）、`days_remain`（剩余的天数，单位为整数，表示为 `days_used` 的值）。

函数的作用是计算剩余的年、月、日和年所占的时间百分比，并返回它。计算方式如下：

1. 计算总天数：将所有参数相乘，然后取整。这个值代表剩余的天数。
2. 计算年数：将剩余的天数除以 365（一年的天数），取整并向下取整。这个值代表剩余的年数。
3. 计算月数：将剩余的天数除以 30（一个月的平均天数），取整并向下取整。这个值代表剩余的月份。
4. 计算日数：将剩余的天数除以 30（一个月的平均天数），取整并向下取整。这个值代表剩余的天数。
5. 返回剩余的年、月、日和年所占的时间百分比：将计算出的年、月、日和年所占的时间百分比分别返回，将它们相加得到总时间百分比，然后将总时间百分比除以 100，得到剩余的时间百分比。

函数的输出结果是一个元组，包含剩余的年、月、日和年所占的时间百分比。


```
def deduct_time(frac, days, years_remain, months_remain, days_remain):
    # CALCULATE TIME IN YEARS, MONTHS, AND DAYS
    days_available = int(frac * days)
    years_used = int(days_available / 365)
    days_available -= years_used * 365
    months_used = int(days_available / 30)
    days_used = days_available - (months_used * 30)
    years_remain = years_remain - years_used
    months_remain = months_remain - months_used
    days_remain = days_remain - days_used

    while days_remain < 0:
        days_remain += 30
        months_remain -= 1

    while months_remain < 0 and years_remain > 0:
        months_remain += 12
        years_remain -= 1
    return years_remain, months_remain, days_remain, years_used, months_used, days_used


```



This code defines three functions, `time_report`, `make_occupation_label`, and `calculate_day_of_week`, which are used to report the number of days in a given year and month, and to label whether the occupation is "Played", "Played/Studied", or "Worked/Played", respectively.

`time_report` function takes four arguments: `msg`, `years`, `months`, and `days`. It first calculates the number of leading spaces needed between the beginning of the `msg` string and the first character, by subtracting the length of the `msg` string from 23. It then prints the string with leading spaces, followed by the `years`, `months`, and `days` in the appropriate format.

`make_occupation_label` function takes a single argument, `years`, and returns a string indicating whether the occupation is "Played", "Played/Studied", or "Worked/Played". This is determined by a simple rule based on the number of years.

`calculate_day_of_week` function takes three arguments: `year`, `month`, and `day`, and returns the day of the week for a given year and month. It does this by initializing a table called `month_table` to store the number of days in each month, and then calculating the appropriate values for the year based on whether it is a leap year or not. It then adds the occupation label (obtained from `make_occupation_label`) and the `days` value to the appropriate cell in the table, and finally calculates the corresponding day of the week by adjusting the weekday value.


```
def time_report(msg, years, months, days):
    leading_spaces = 23 - len(msg)
    print(" " * leading_spaces + f"{msg}\t{years}\t{months}\t{days}")


def make_occupation_label(years):
    if years <= 3:
        return "PLAYED"
    elif years <= 9:
        return "PLAYED/STUDIED"
    else:
        return "WORKED/PLAYED"


def calculate_day_of_week(year, month, day):
    # Initial values for months
    month_table = [0, 3, 3, 6, 1, 4, 6, 2, 5, 0, 3, 5]

    i1 = int((year - 1500) / 100)
    a = i1 * 5 + (i1 + 3) / 4
    i2 = int(a - int(a / 7) * 7)
    y2 = int(year / 100)
    y3 = int(year - y2 * 100)
    a = y3 / 4 + y3 + day + month_table[month - 1] + i2
    b = int(a - int(a / 7) * 7) + 1
    b = adjust_weekday(b, month, year)

    return b


```

This appears to be a Python code that generates a table with some data and also a table with some information about the user's age and the number of days they have spent on different activities like sleeping, eating, and relaxing.

It uses a function called "deduct\_time" to calculate the number of days the user has spent on these activities based on their age and the number of days they have lived.

It also includes some code that calculates the retirement date based on the user's age and the number of years they have lived.

It is using a variable "el\_years" that is assumed to be the user's birth date and "el\_months" and "el\_days" that are calculated based on the number of days the user has lived.


```
def end() -> None:
    for _ in range(5):
        print()


def main() -> None:
    print(" " * 32 + "WEEKDAY")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    print("WEEKDAY IS A COMPUTER DEMONSTRATION THAT")
    print("GIVES FACTS ABOUT A DATE OF INTEREST TO YOU.")
    print()

    if GET_TODAY_FROM_SYSTEM:
        month_today, day_today, year_today = get_date_from_system()
    else:
        month_today, day_today, year_today = get_date_from_user(
            "ENTER TODAY'S DATE IN THE FORM: 3,24,1979"
        )

    # This program determines the day of the week
    # for a date after 1582

    print()

    month, day, year = get_date_from_user(
        "ENTER DAY OF BIRTH (OR OTHER DAY OF INTEREST) (like MM,DD,YYYY)"
    )

    print()

    # Test for date before current calendar
    if year < 1582:
        print("NOT PREPARED TO GIVE DAY OF WEEK PRIOR TO MDLXXXII.")
        end()
        return

    b = calculate_day_of_week(year, month, day)

    today_day_value = calc_day_value(year_today, month_today, day_today)
    target_day_value = calc_day_value(year, month, day)

    is_today = False

    if today_day_value < target_day_value:
        label = "WILL BE A"
    elif today_day_value == target_day_value:
        label = "IS A"
        is_today = True
    else:
        label = "WAS A"

    day_name = get_day_of_week(b, day)

    # print the day of the week the date falls on.
    print(f"{month}/{day}/{year} {label} {day_name}.")

    if is_today:
        # nothing to report for today
        end()
        return

    print()

    el_years = year_today - year
    el_months = month_today - month
    el_days = day_today - day

    if el_days < 0:
        el_months = el_months - 1
        el_days = el_days + 30
    if el_months < 0:
        el_years = el_years - 1
        el_months = el_months + 12
    if el_years < 0:
        # target date is in the future
        end()
        return

    if (el_months == 0) and (el_days == 0):
        print("***HAPPY BIRTHDAY***")

    # print report
    print(" " * 23 + "\tYEARS\tMONTHS\tDAYS")
    print(" " * 23 + "\t-----\t------\t----")
    print(f"YOUR AGE (IF BIRTHDATE)\t{el_years}\t{el_months}\t{el_days}")

    life_days = (el_years * 365) + (el_months * 30) + el_days + int(el_months / 2)
    rem_years = el_years
    rem_months = el_months
    rem_days = el_days

    rem_years, rem_months, rem_days, used_years, used_months, used_days = deduct_time(
        0.35, life_days, rem_years, rem_months, rem_days
    )
    time_report("YOU HAVE SLEPT", used_years, used_months, used_days)
    rem_years, rem_months, rem_days, used_years, used_months, used_days = deduct_time(
        0.17, life_days, rem_years, rem_months, rem_days
    )
    time_report("YOU HAVE EATEN", used_years, used_months, used_days)

    label = make_occupation_label(rem_years)
    rem_years, rem_months, rem_days, used_years, used_months, used_days = deduct_time(
        0.23, life_days, rem_years, rem_months, rem_days
    )
    time_report("YOU HAVE " + label, used_years, used_months, used_days)
    time_report("YOU HAVE RELAXED", rem_years, rem_months, rem_days)

    print()

    # Calculate retirement date
    e = year + 65
    print(" " * 16 + f"***  YOU MAY RETIRE IN {e} ***")
    end()


```

这段代码是一个条件判断语句，它会判断当前脚本是否作为主程序运行。如果是主程序运行，那么程序会执行if语句块内的内容。

具体来说，这段代码的意义是：如果当前脚本作为主程序运行，那么执行if语句块内的内容。否则，不执行if语句块内的内容。

如果这个条件不满足，这段代码也不会输出任何内容。


```
if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Rust](https://www.rust-lang.org/) by Anthony Rubick [AnthonyMichaelTDM](https://github.com/AnthonyMichaelTDM)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)


### Word

WORD is a combination of HANGMAN and BAGELS. In this game, the player must guess a word with clues as to a letter position furnished by the computer. However, instead of guessing one letter at a time, in WORD you guess an entire word (or group of 5 letters, such as ABCDE). The computer will tell you if any letters that you have guessed are in the mystery word and if any of them are in the correct position. Armed with these clues, you go on guessing until you get the word or, if you can’t get it, input a “?” and the computer will tell you the mystery word.

You may change the words in Data Statements, but they must be 5-letter words.

The author of this program is Charles Reid of Lexington High School, Lexington, Massachusetts.

---

As published in Basic Computer Games (1978):
- [Atari Archives](https://www.atariarchives.org/basicgames/showpage.php?page=181)
- [Annarchive](https://annarchive.com/files/Basic_Computer_Games_Microcomputer_Edition.pdf#page=194)

Downloaded from Vintage Basic at
http://www.vintage-basic.net/games.html

#### Porting Notes

(please note any difficulties or challenges in porting here)


# `96_Word/csharp/Program.cs`

This looks like a simple game where the user has to guess a secret word that is given by the Word class. The Word class has a list of words that it will pick one from, and a method to check if the user's guess is correct or not. The user will have to keep guessing until they either correctly guess the word or they give up.

Is there anything else you'd like to know about this code?


```
﻿using System;
using System.Linq;
using System.Text;

namespace word
{
    class Word
    {
        // Here's the list of potential words that could be selected
        // as the winning word.
        private string[] words = { "DINKY", "SMOKE", "WATER", "GRASS", "TRAIN", "MIGHT", "FIRST",
         "CANDY", "CHAMP", "WOULD", "CLUMP", "DOPEY" };

        /// <summary>
        /// Outputs the instructions of the game.
        /// </summary>
        private void intro()
        {
            Console.WriteLine("WORD".PadLeft(37));
            Console.WriteLine("CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY".PadLeft(59));

            Console.WriteLine("I am thinking of a word -- you guess it. I will give you");
            Console.WriteLine("clues to help you get it. Good luck!!");
        }

        /// <summary>
        /// This allows the user to enter a guess - doing some basic validation
        /// on those guesses.
        /// </summary>
        /// <returns>The guess entered by the user</returns>
        private string get_guess()
        {
            string guess = "";

            while (guess.Length == 0)
            {
                Console.WriteLine($"{Environment.NewLine}Guess a five letter word. ");
                guess = Console.ReadLine().ToUpper();

                if ((guess.Length != 5) || (guess.Equals("?")) || (!guess.All(char.IsLetter)))
                {
                    guess = "";
                    Console.WriteLine("You must guess a five letter word. Start again.");
                }
            }

            return guess;
        }

        /// <summary>
        /// This checks the user's guess against the target word - capturing
        /// any letters that match up between the two as well as the specific
        /// letters that are correct.
        /// </summary>
        /// <param name="guess">The user's guess</param>
        /// <param name="target">The 'winning' word</param>
        /// <param name="progress">A string showing which specific letters have already been guessed</param>
        /// <returns>The integer value showing the number of character matches between guess and target</returns>
        private int check_guess(string guess, string target, StringBuilder progress)
        {
            // Go through each letter of the guess and see which
            // letters match up to the target word.
            // For each position that matches, update the progress
            // to reflect the guess
            int matches = 0;
            string common_letters = "";

            for (int ctr = 0; ctr < 5; ctr++)
            {
                // First see if this letter appears anywhere in the target
                // and, if so, add it to the common_letters list.
                if (target.Contains(guess[ctr]))
                {
                    common_letters.Append(guess[ctr]);
                }
                // Then see if this specific letter matches the
                // same position in the target. And, if so, update
                // the progress tracker
                if (guess[ctr].Equals(target[ctr]))
                {
                    progress[ctr] = guess[ctr];
                    matches++;
                }
            }

            Console.WriteLine($"There were {matches} matches and the common letters were... {common_letters}");
            Console.WriteLine($"From the exact letter matches, you know......... {progress}");
            return matches;
        }

        /// <summary>
        /// This plays one full game.
        /// </summary>
        private void play_game()
        {
            string guess_word, target_word;
            StringBuilder guess_progress = new StringBuilder("-----");
            Random rand = new Random();
            int count = 0;

            Console.WriteLine("You are starting a new game...");

            // Randomly select a word from the list of words
            target_word = words[rand.Next(words.Length)];

            // Just run as an infinite loop until one of the
            // endgame conditions are met.
            while (true)
            {
                // Ask the user for their guess
                guess_word = get_guess();
                count++;

                // If they enter a question mark, then tell them
                // the answer and quit the game
                if (guess_word.Equals("?"))
                {
                    Console.WriteLine($"The secret word is {target_word}");
                    return;
                }

                // Otherwise, check the guess against the target - noting progress
                if (check_guess(guess_word, target_word, guess_progress) == 0)
                {
                    Console.WriteLine("If you give up, type '?' for your next guess.");
                }

                // Once they've guess the word, end the game.
                if (guess_progress.Equals(guess_word))
                {
                    Console.WriteLine($"You have guessed the word.  It took {count} guesses!");
                    return;
                }
            }
        }

        /// <summary>
        /// The main entry point for the class - just keeps
        /// playing the game until the user decides to quit.
        /// </summary>
        public void play()
        {
            intro();

            bool keep_playing = true;

            while (keep_playing)
            {
                play_game();
                Console.WriteLine($"{Environment.NewLine}Want to play again? ");
                keep_playing = Console.ReadLine().StartsWith("y", StringComparison.CurrentCultureIgnoreCase);
            }
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            new Word().play();
        }
    }
}

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Microsoft C#](https://docs.microsoft.com/en-us/dotnet/csharp/)


Original source downloaded from [Vintage Basic](http://www.vintage-basic.net/games.html)

Converted to [D](https://dlang.org/) by [Bastiaan Veelo](https://github.com/veelo).

The Basic original required words to be exactly five letters in length for the program to behave correctly.
This version does not replicate that limitation, and the test for that requirement is commented out.

## Running the code

Assuming the reference [dmd](https://dlang.org/download.html#dmd) compiler:
```shell
dmd -dip1000 -run word.d
```

[Other compilers](https://dlang.org/download.html) also exist.


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Oracle Java](https://openjdk.java.net/)


# `96_Word/java/Word.java`

该代码的作用是实现了一个基于1970年代BASIC游戏Word的Java版本。这个程序通过从用户那里获取单词列表来进行游戏。


```
import java.util.Arrays;
import java.util.Scanner;

/**
 * Game of Word
 * <p>
 * Based on the BASIC game of Word here
 * https://github.com/coding-horror/basic-computer-games/blob/main/96%20Word/word.bas
 * <p>
 * Note:  The idea was to create a version of the 1970's BASIC game in Java, without introducing
 * new features - no additional text, error checking, etc has been added.
 *
 * Converted from BASIC to Java by Darren Cardenas.
 */

```

This is a Java program that allows the user to guess a secret word by giving hints. The program uses a while loop with an explicit退出 condition to keep the user from guessing indefinitely.

The secret word is stored in a variable called `word`, and the program uses several helper methods to manipulate the text string and perform operations on the input.

The main method first initializes the `word` object and then enters a while loop that runs until the user decides to exit the game. In each iteration of the loop, the program checks for various events that may indicate the user has found a match or reached a milestone.

If the user enters the name of the secret word, the program displays a win message and then prompts the user to enter their name again to continue playing. If the user does not enter the name, the program will exit.

If the user makes a guess and the number of guesses is less than or equal to the number of matches, the program displays a win message and then prompts the user to enter their name again to continue playing. If the user makes more than one guess, the program displays a lose message and then ends the game.

If the while loop completes and the user has not left the game after that, the program displays a win message and ends the game.

The program also includes a helper method called `makeLegalWord(String)` to convert a word string to lowercase and remove any non-alphanumeric characters.


```
public class Word {

  private final static String[] WORDS = {

  "DINKY", "SMOKE", "WATER", "GRASS", "TRAIN", "MIGHT",
  "FIRST", "CANDY", "CHAMP", "WOULD", "CLUMP", "DOPEY"

  };

  private final Scanner scan;  // For user input

  private enum Step {
    INITIALIZE, MAKE_GUESS, USER_WINS
  }

  public Word() {

    scan = new Scanner(System.in);

  }  // End of constructor Word

  public void play() {

    showIntro();
    startGame();

  }  // End of method play

  private void showIntro() {

    System.out.println(" ".repeat(32) + "WORD");
    System.out.println(" ".repeat(14) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
    System.out.println("\n\n");

    System.out.println("I AM THINKING OF A WORD -- YOU GUESS IT.  I WILL GIVE YOU");
    System.out.println("CLUES TO HELP YOU GET IT.  GOOD LUCK!!");
    System.out.println("\n");

  }  // End of method showIntro

  private void startGame() {

    char[] commonLetters = new char[8];
    char[] exactLetters = new char[8];

    int commonIndex = 0;
    int ii = 0;  // Loop iterator
    int jj = 0;  // Loop iterator
    int numGuesses = 0;
    int numMatches = 0;
    int wordIndex = 0;

    Step nextStep = Step.INITIALIZE;

    String commonString = "";
    String exactString = "";
    String guessWord = "";
    String secretWord = "";
    String userResponse = "";

    // Begin outer while loop
    while (true) {

      switch (nextStep) {

        case INITIALIZE:

          System.out.println("\n");
          System.out.println("YOU ARE STARTING A NEW GAME...");

          // Select a secret word from the list
          wordIndex = (int) (Math.random() * WORDS.length);
          secretWord = WORDS[wordIndex];

          numGuesses = 0;

          Arrays.fill(exactLetters, 1, 6, '-');
          Arrays.fill(commonLetters, 1, 6, '\0');

          nextStep = Step.MAKE_GUESS;
          break;

        case MAKE_GUESS:

          System.out.print("GUESS A FIVE LETTER WORD? ");
          guessWord = scan.nextLine().toUpperCase();

          numGuesses++;

          // Win condition
          if (guessWord.equals(secretWord)) {
            nextStep = Step.USER_WINS;
            continue;
          }

          Arrays.fill(commonLetters, 1, 8, '\0');

          // Surrender condition
          if (guessWord.equals("?")) {
            System.out.println("THE SECRET WORD IS " + secretWord);
            System.out.println("");
            nextStep = Step.INITIALIZE;  // Play again
            continue;
          }

          // Check for valid input
          if (guessWord.length() != 5) {
            System.out.println("YOU MUST GUESS A 5 LETTER WORD.  START AGAIN.");
            numGuesses--;
            nextStep = Step.MAKE_GUESS;  // Guess again
            continue;
          }

          numMatches = 0;
          commonIndex = 1;

          for (ii = 1; ii <= 5; ii++) {

            for (jj = 1; jj <= 5; jj++) {

              if (secretWord.charAt(ii - 1) != guessWord.charAt(jj - 1)) {
                continue;
              }

              // Avoid out of bounds errors
              if (commonIndex <= 5) {
                commonLetters[commonIndex] = guessWord.charAt(jj - 1);
                commonIndex++;
              }

              if (ii == jj) {
                exactLetters[jj] = guessWord.charAt(jj - 1);
              }

              // Avoid out of bounds errors
              if (numMatches < 5) {
                numMatches++;
              }
            }
          }

          exactString = "";
          commonString = "";

          // Build the exact letters string
          for (ii = 1; ii <= 5; ii++) {
            exactString += exactLetters[ii];
          }

          // Build the common letters string
          for (ii = 1; ii <= numMatches; ii++) {
            commonString += commonLetters[ii];
          }

          System.out.println("THERE WERE " + numMatches + " MATCHES AND THE COMMON LETTERS WERE..."
                             + commonString);

          System.out.println("FROM THE EXACT LETTER MATCHES, YOU KNOW................" + exactString);

          // Win condition
          if (exactString.equals(secretWord)) {
            nextStep = Step.USER_WINS;
            continue;
          }

          // No matches
          if (numMatches <= 1) {
            System.out.println("");
            System.out.println("IF YOU GIVE UP, TYPE '?' FOR YOUR NEXT GUESS.");
          }

          System.out.println("");
          nextStep = Step.MAKE_GUESS;
          break;

        case USER_WINS:

          System.out.println("YOU HAVE GUESSED THE WORD.  IT TOOK " + numGuesses + " GUESSES!");
          System.out.println("");

          System.out.print("WANT TO PLAY AGAIN? ");
          userResponse = scan.nextLine();

          if (userResponse.toUpperCase().equals("YES")) {
            nextStep = Step.INITIALIZE;  // Play again
          } else {
            return;  // Quit game
          }
          break;

        default:
          System.out.println("INVALID STEP");
          break;

      }

    }  // End outer while loop

  }  // End of method startGame

  public static void main(String[] args) {

    Word word = new Word();
    word.play();

  }  // End of method main

}  // End of class Word

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Shells)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Kotlin](https://kotlinlang.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Lua](https://www.lua.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Perl](https://www.perl.org/)


Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Python](https://www.python.org/about/)


# `96_Word/python/word.py`

这段代码是一个Python脚本，它将文本字符串“WORDS”转换为随机的单词。

具体来说，它首先导入了一个名为“random”的模块，然后定义了一个包含多个单词列表的列表“words”。接下来，它使用列表推导式从“words”列表中遍历每个元素，并为每个元素生成一个随机单词。最后，它将生成的单词打印出来。

因此，这段代码的作用是生成一系列随机的单词，并将它们打印出来。


```
#!/usr/bin/env python3

"""
WORD

Converted from BASIC to Python by Trevor Hobson
"""

import random

words = [
    "DINKY",
    "SMOKE",
    "WATER",
    "GRASS",
    "TRAIN",
    "MIGHT",
    "FIRST",
    "CANDY",
    "CHAMP",
    "WOULD",
    "CLUMP",
    "DOPEY",
]


```

这段代码是一个 Python 函数，名为 `play_game()`，它用于在给定游戏中进行一次游戏。在这个游戏中，玩家需要猜测一个 5 字母的单词，并尝试找到与游戏中的“秘密单词”相同的单词。

函数内部首先从给定的单词列表中随机选择一个元素，然后玩家需要输入一个 5 字母的单词作为答案。如果玩家输入的单词是“？”则函数会要求玩家再次输入。如果输入的单词是正确的，则函数会告诉玩家他们猜对了密码，并结束游戏。否则，函数会告诉玩家他们猜错了密码，然后要求玩家继续猜测。

函数内部会记录每个玩家猜测的单词数量，并在每次猜测失败时告诉玩家他们猜错了密码。如果玩家在猜测中猜中了密码，则函数会告诉玩家他们猜对了密码，并结束游戏。如果玩家猜测了错误的密码，则会提示玩家继续尝试。


```
def play_game() -> None:
    """Play one round of the game"""

    random.shuffle(words)
    target_word = words[0]
    guess_count = 0
    guess_progress = ["-"] * 5

    print("You are starting a new game...")
    while True:
        guess_word = ""
        while guess_word == "":
            guess_word = input("\nGuess a five letter word. ").upper()
            if guess_word == "?":
                break
            elif not guess_word.isalpha() or len(guess_word) != 5:
                guess_word = ""
                print("You must guess a five letter word. Start again.")
        guess_count += 1
        if guess_word == "?":
            print("The secret word is", target_word)
            break
        else:
            common_letters = ""
            matches = 0
            for i in range(5):
                for j in range(5):
                    if guess_word[i] == target_word[j]:
                        matches += 1
                        common_letters = common_letters + guess_word[i]
                        if i == j:
                            guess_progress[j] = guess_word[i]
            print(
                f"There were {matches}",
                f"matches and the common letters were... {common_letters}",
            )
            print(
                "From the exact letter matches, you know............ "
                + "".join(guess_progress)
            )
            if "".join(guess_progress) == guess_word:
                print(f"\nYou have guessed the word. It took {guess_count} guesses!")
                break
            elif matches == 0:
                print("\nIf you give up, type '?' for you next guess.")


```

这段代码是一个Python程序，名为“main”。程序的主要目的是让用户玩一个猜词游戏。程序会随机生成一个单词，然后给出几个提示，让用户猜测生词。程序会不断地询问用户是否想继续玩游戏，如果用户输入“y”，程序就会继续。

具体来说，这段代码包含以下几个部分：

1. 定义了一个名为“main”的函数，它接受一个空括号作为参数，然后返回一个空括号。这是Python中定义函数的标准方式。
2. 在函数内部，程序首先输出一段白色的空间，并在其中添加了一些“单词”的字样。然后，程序会输出一段绿色的空间，并在其中添加了一些“创造性的计算”的字样，这是在告诉用户这是一个创造性的计算环境。
3. 程序接下来输出一段白色的空间，并在其中插入了一个问号，然后提示用户猜测生词。
4. 程序接下来会循环，直到用户输入“Want to play again?”来停止循环。如果用户输入“y”，程序就会继续循环；如果用户输入其他字符（如空格、回车等），程序就会退出循环并退出程序。
5. 程序最后定义了一个名为“keep_playing”的变量，并将其设置为True。在循环的每次迭代中，程序都会调用一个名为“play_game”的函数，但这个函数的具体实现不在本次代码中。


```
def main() -> None:
    print(" " * 33 + "WORD")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n")

    print("I am thinking of a word -- you guess it. I will give you")
    print("clues to help you get it. Good luck!!\n")

    keep_playing = True
    while keep_playing:
        play_game()
        keep_playing = input("\nWant to play again? ").lower().startswith("y")


if __name__ == "__main__":
    main()

```

Original source downloaded [from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Ruby](https://www.ruby-lang.org/en/)


Original BASIC source [downloaded from Vintage Basic](http://www.vintage-basic.net/games.html)

Conversion to [Visual Basic .NET](https://en.wikipedia.org/wiki/Visual_Basic_.NET)
