# `basic-computer-games\21_Calendar\python\calendar.py`

```py
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
        # 根据输入的星期几，确定初始星期几的数字表示
        for day_k in days_mapping.keys():
            if weekday.lower() in day_k:
                day = days_mapping[day_k]
                correct_day_input = True
                break

    while True:
        leap = input("IS IT A LEAP YEAR?:")
        # 判断是否是闰年
        if "y" in leap.lower():
            leap_day = True
            break

        if "n" in leap.lower():
            leap_day = False
            break

    return day, leap_day


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
    # 如果是闰年，则将二月的天数修改为29
    if leap_year:
        months_days[2] = 29
        years_day = 366

    # 月份名称列表
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

    # 主循环
    for n in range(1, 13):
        # 累加每个月的天数
        days_count += months_days[n - 1]
        # 打印月份信息和剩余天数
        print(
            f"** {days_count} ****************** {months_names[n - 1]} "
            f"****************** {years_day - days_count} **\n"
        )
        print(days)
        print(sep)

        # 打印日历
        for _ in range(1, 7):
            print("\n")
            for g in range(1, 8):  # noqa
                d += 1
                d2 = d - days_count

                # 如果超过了本月的天数，则跳出循环
                if d2 > months_days[n]:
                    break

                # 如果日期小于等于0，则打印空格
                if d2 <= 0:
                    print("  ", end="       ")
                # 如果日期小于10，则在前面补空格
                elif d2 < 10:
                    print(f" {d2}", end="       ")
                else:
                    print(f"{d2}", end="       ")
            print()

            # 如果日期超过了本月的天数，则跳出循环
            if d2 >= months_days[n]:
                break

        # 如果日期超过了本月的天数，则回退日期
        if d2 > months_days[n]:
            d -= g

        print("\n")

    print("\n")
# 定义主函数
def main() -> None:
    # 打印日历标题
    print(" " * 32 + "CALENDAR")
    # 打印创意计算的地点
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")
    # 打印11个空行
    print("\n" * 11)

    # 解析用户输入的第一天和是否是闰年
    day, leap_year = parse_input()
    # 调用日历函数
    calendar(day, leap_year)


# 如果当前脚本被直接执行，则调用主函数
if __name__ == "__main__":
    main()

########################################################
#
# 移植说明:
#
# 在程序开头添加了一个输入，用户可以指定一年中的第一天是星期几，以及这一年是否是闰年。
#
########################################################
```