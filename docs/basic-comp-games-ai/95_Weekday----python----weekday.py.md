# `basic-computer-games\95_Weekday\python\weekday.py`

```py
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

GET_TODAY_FROM_SYSTEM = True


def get_date_from_user(prompt: str) -> Tuple[int, int, int]:
    # 从用户输入获取日期
    while True:
        print(prompt)
        date_str = input()
        try:
            # 将用户输入的日期字符串分割成月、日、年，并返回元组
            month_num, day_num, year_num = (int(x) for x in date_str.split(","))
            return month_num, day_num, year_num
        except Exception:
            print("I COULDN'T UNDERSTAND THAT. TRY AGAIN.")


def get_date_from_system() -> Tuple[int, int, int]:
    # 从系统获取当前日期
    dt = datetime.datetime.today()
    return dt.month, dt.day, dt.year


def get_day_of_week(weekday_index, day) -> str:
    # 返回对应的星期几名称
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


def previous_day(b) -> int:
    # 返回前一天的索引
    if b == 0:
        b = 6
    return b - 1


def is_leap_year(year: int) -> bool:
    # 判断是否为闰年
    if (year % 4) != 0:
        return False
    if (year % 100) != 0:
        return True
    if (year % 400) != 0:
        return False
    return True


def adjust_day_for_leap_year(b, year):
    # 如果是闰年，调整索引
    if is_leap_year(year):
        b = previous_day(b)
    return b


def adjust_weekday(b, month, year):
    # 调整索引以适应闰年
    if month <= 2:
        b = adjust_day_for_leap_year(b, year)
    if b == 0:
        b = 7
    return b


def calc_day_value(year, month, day):
    # 计算日期的值
    return (year * 12 + month) * 31 + day


def deduct_time(frac, days, years_remain, months_remain, days_remain):
    # CALCULATE TIME IN YEARS, MONTHS, AND DAYS
    # 计算年、月、日的时间
    days_available = int(frac * days)
    years_used = int(days_available / 365)
    # 计算剩余天数
    days_available -= years_used * 365
    # 计算已使用的月数
    months_used = int(days_available / 30)
    # 计算已使用的天数
    days_used = days_available - (months_used * 30)
    # 计算剩余年数
    years_remain = years_remain - years_used
    # 计算剩余月数
    months_remain = months_remain - months_used
    # 计算剩余天数
    days_remain = days_remain - days_used

    # 处理剩余天数小于0的情况
    while days_remain < 0:
        days_remain += 30
        months_remain -= 1

    # 处理剩余月数小于0且剩余年数大于0的情况
    while months_remain < 0 and years_remain > 0:
        months_remain += 12
        years_remain -= 1
    # 返回剩余的年月日和已使用的年月日
    return years_remain, months_remain, days_remain, years_used, months_used, days_used
# 打印时间报告
def time_report(msg, years, months, days):
    # 计算前导空格数，使得消息和时间对齐
    leading_spaces = 23 - len(msg)
    # 打印消息和时间
    print(" " * leading_spaces + f"{msg}\t{years}\t{months}\t{days}")


# 生成职业标签
def make_occupation_label(years):
    if years <= 3:
        return "PLAYED"
    elif years <= 9:
        return "PLAYED/STUDIED"
    else:
        return "WORKED/PLAYED"


# 计算星期几
def calculate_day_of_week(year, month, day):
    # 月份初始值表
    month_table = [0, 3, 3, 6, 1, 4, 6, 2, 5, 0, 3, 5]

    i1 = int((year - 1500) / 100)
    a = i1 * 5 + (i1 + 3) / 4
    i2 = int(a - int(a / 7) * 7)
    y2 = int(year / 100)
    y3 = int(year - y2 * 100)
    a = y3 / 4 + y3 + day + month_table[month - 1] + i2
    b = int(a - int(a / 7) * 7) + 1
    b = adjust_weekday(b, month, year)  # 调整星期几
    return b


# 结束函数
def end() -> None:
    # 打印空行
    for _ in range(5):
        print()


# 主函数
def main() -> None:
    # 打印标题
    print(" " * 32 + "WEEKDAY")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    print("WEEKDAY IS A COMPUTER DEMONSTRATION THAT")
    print("GIVES FACTS ABOUT A DATE OF INTEREST TO YOU.")
    print()

    # 获取今天的日期
    if GET_TODAY_FROM_SYSTEM:
        month_today, day_today, year_today = get_date_from_system()
    else:
        month_today, day_today, year_today = get_date_from_user(
            "ENTER TODAY'S DATE IN THE FORM: 3,24,1979"
        )

    # 这个程序确定1582年后的日期是星期几
    print()

    # 获取用户输入的日期
    month, day, year = get_date_from_user(
        "ENTER DAY OF BIRTH (OR OTHER DAY OF INTEREST) (like MM,DD,YYYY)"
    )

    print()

    # 测试日期是否在当前日历之前
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
    # 如果今天的日期值等于目标日期值
    elif today_day_value == target_day_value:
        # 设置标签为"IS A"
        label = "IS A"
        # 设置is_today为True
        is_today = True
    # 如果不是上述情况
    else:
        # 设置标签为"WAS A"
        label = "WAS A"

    # 获取日期对应的星期几的名称
    day_name = get_day_of_week(b, day)

    # 打印日期所在的星期几
    print(f"{month}/{day}/{year} {label} {day_name}.")

    # 如果是今天的日期
    if is_today:
        # 今天没有需要报告的内容
        end()
        return

    # 打印空行
    print()

    # 计算年龄
    el_years = year_today - year
    el_months = month_today - month
    el_days = day_today - day

    # 如果天数小于0
    if el_days < 0:
        el_months = el_months - 1
        el_days = el_days + 30
    # 如果月数小于0
    if el_months < 0:
        el_years = el_years - 1
        el_months = el_months + 12
    # 如果年数小于0
    if el_years < 0:
        # 目标日期在未来
        end()
        return

    # 如果月数和天数都为0
    if (el_months == 0) and (el_days == 0):
        # 打印生日祝福
        print("***HAPPY BIRTHDAY***")

    # 打印报告表头
    print(" " * 23 + "\tYEARS\tMONTHS\tDAYS")
    print(" " * 23 + "\t-----\t------\t----")
    print(f"YOUR AGE (IF BIRTHDATE)\t{el_years}\t{el_months}\t{el_days}")

    # 计算生活天数
    life_days = (el_years * 365) + (el_months * 30) + el_days + int(el_months / 2)
    rem_years = el_years
    rem_months = el_months
    rem_days = el_days

    # 减去睡眠时间
    rem_years, rem_months, rem_days, used_years, used_months, used_days = deduct_time(
        0.35, life_days, rem_years, rem_months, rem_days
    )
    time_report("YOU HAVE SLEPT", used_years, used_months, used_days)
    # 减去进食时间
    rem_years, rem_months, rem_days, used_years, used_months, used_days = deduct_time(
        0.17, life_days, rem_years, rem_months, rem_days
    )
    time_report("YOU HAVE EATEN", used_years, used_months, used_days)

    # 根据剩余年限生成职业标签
    label = make_occupation_label(rem_years)
    # 减去工作时间
    rem_years, rem_months, rem_days, used_years, used_months, used_days = deduct_time(
        0.23, life_days, rem_years, rem_months, rem_days
    )
    time_report("YOU HAVE " + label, used_years, used_months, used_days)
    # 减去休闲时间
    time_report("YOU HAVE RELAXED", rem_years, rem_months, rem_days)

    # 打印空行
    print()

    # 计算退休日期
    # 计算退休年龄
    e = year + 65
    # 打印提示信息，使用 f-string 格式化输出退休年份
    print(" " * 16 + f"***  YOU MAY RETIRE IN {e} ***")
    # 调用 end() 函数
    end()
# 如果当前模块被直接执行，则调用 main() 函数
if __name__ == "__main__":
    main()
```