# `d:/src/tocomm/basic-computer-games\95_Weekday\python\weekday.py`

```
"""
WEEKDAY

Calculates which weekday an entered date is.

Also estimates how long a person has done certain activities, if they
entered their birthday.

Also calculates the year of retirement, assuming retiring at age 65.

Ported by Dave LeCompte.
"""

import datetime  # 导入 datetime 模块
from typing import Tuple  # 从 typing 模块导入 Tuple 类型

GET_TODAY_FROM_SYSTEM = True  # 设置获取今天日期的标志为 True


def get_date_from_user(prompt: str) -> Tuple[int, int, int]:  # 定义一个函数，接受一个字符串参数，返回一个包含三个整数的元组
    while True:  # 无限循环，直到条件被打破
        print(prompt)  # 打印提示信息
        date_str = input()  # 获取用户输入的日期字符串
        try:  # 尝试执行以下代码
            month_num, day_num, year_num = (int(x) for x in date_str.split(","))  # 将用户输入的日期字符串按逗号分隔，并转换为整数
            return month_num, day_num, year_num  # 返回月份、日期和年份
        except Exception:  # 如果出现异常
            print("I COULDN'T UNDERSTAND THAT. TRY AGAIN.")  # 打印错误信息
            

def get_date_from_system() -> Tuple[int, int, int]:  # 定义一个函数，返回类型为元组，包含三个整数
    dt = datetime.datetime.today()  # 获取当前日期时间
    return dt.month, dt.day, dt.year  # 返回月份、日期和年份


def get_day_of_week(weekday_index, day) -> str:  # 定义一个函数，接受两个参数，返回类型为字符串
    day_names = {  # 创建一个包含星期索引和对应名称的字典
        1: "SUNDAY",
        2: "MONDAY",
        3: "TUESDAY",
        4: "WEDNESDAY",  # 定义星期四对应的索引值为4
        5: "THURSDAY",    # 定义星期五对应的索引值为5
        6: "FRIDAY",      # 定义星期六对应的索引值为6
        7: "SATURDAY",    # 定义星期日对应的索引值为7
    }

    if weekday_index == 6 and day == 13:  # 如果星期索引为6（即星期六）且日期为13
        return "FRIDAY THE THIRTEENTH---BEWARE!"  # 返回“星期五13日---当心！”
    return day_names[weekday_index]  # 返回对应星期索引的星期名称


def previous_day(b) -> int:  # 定义函数previous_day，参数为b，返回类型为整数
    if b == 0:  # 如果b等于0
        b = 6  # 将b赋值为6
    return b - 1  # 返回b减去1的结果


def is_leap_year(year: int) -> bool:  # 定义函数is_leap_year，参数为year，返回类型为布尔值
    if (year % 4) != 0:  # 如果year除以4的余数不等于0
        return False  # 返回False，表示不是闰年
    if (year % 100) != 0:  # 如果年份不是整百年
        return True  # 返回True
    if (year % 400) != 0:  # 如果年份不是整百年但是整400年
        return False  # 返回False
    return True  # 其他情况返回True


def adjust_day_for_leap_year(b, year):
    if is_leap_year(year):  # 如果是闰年
        b = previous_day(b)  # 调整日期为前一天
    return b  # 返回调整后的日期


def adjust_weekday(b, month, year):
    if month <= 2:  # 如果月份在1或2月
        b = adjust_day_for_leap_year(b, year)  # 调整日期为闰年的情况
    if b == 0:  # 如果日期是星期天
        b = 7  # 调整为星期天的表示方式
    return b  # 返回调整后的星期几
# 计算给定日期的数值表示
def calc_day_value(year, month, day):
    return (year * 12 + month) * 31 + day


# 扣除时间
def deduct_time(frac, days, years_remain, months_remain, days_remain):
    # 计算年、月和日
    days_available = int(frac * days)  # 计算可用天数
    years_used = int(days_available / 365)  # 计算使用的年数
    days_available -= years_used * 365  # 减去已使用的年数后剩余的天数
    months_used = int(days_available / 30)  # 计算使用的月数
    days_used = days_available - (months_used * 30)  # 减去已使用的月数后剩余的天数
    years_remain = years_remain - years_used  # 减去已使用的年数后剩余的年数
    months_remain = months_remain - months_used  # 减去已使用的月数后剩余的月数
    days_remain = days_remain - days_used  # 减去已使用的天数后剩余的天数

    # 处理天数为负数的情况
    while days_remain < 0:
        days_remain += 30
        months_remain -= 1
    while months_remain < 0 and years_remain > 0:  # 当月份剩余小于0且年份剩余大于0时执行循环
        months_remain += 12  # 将月份剩余加上12
        years_remain -= 1  # 年份剩余减去1
    return years_remain, months_remain, days_remain, years_used, months_used, days_used  # 返回年份剩余、月份剩余、天数剩余、已使用年份、已使用月份、已使用天数


def time_report(msg, years, months, days):  # 定义一个函数，用于输出时间报告
    leading_spaces = 23 - len(msg)  # 计算前导空格的数量
    print(" " * leading_spaces + f"{msg}\t{years}\t{months}\t{days}")  # 输出消息、年份、月份、天数


def make_occupation_label(years):  # 定义一个函数，用于生成职业标签
    if years <= 3:  # 如果年份小于等于3
        return "PLAYED"  # 返回"PLAYED"
    elif years <= 9:  # 如果年份小于等于9
        return "PLAYED/STUDIED"  # 返回"PLAYED/STUDIED"
    else:  # 否则
        return "WORKED/PLAYED"  # 返回"WORKED/PLAYED"
def calculate_day_of_week(year, month, day):
    # 初始化月份表
    month_table = [0, 3, 3, 6, 1, 4, 6, 2, 5, 0, 3, 5]

    # 计算公式中的各个参数
    i1 = int((year - 1500) / 100)
    a = i1 * 5 + (i1 + 3) / 4
    i2 = int(a - int(a / 7) * 7)
    y2 = int(year / 100)
    y3 = int(year - y2 * 100)
    a = y3 / 4 + y3 + day + month_table[month - 1] + i2
    b = int(a - int(a / 7) * 7) + 1
    # 调整星期几的值
    b = adjust_weekday(b, month, year)

    return b


def end() -> None:
    # 打印5行空行
    for _ in range(5):
        print()
def main() -> None:
    # 打印标题
    print(" " * 32 + "WEEKDAY")
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n\n\n")
    print("WEEKDAY IS A COMPUTER DEMONSTRATION THAT")
    print("GIVES FACTS ABOUT A DATE OF INTEREST TO YOU.")
    print()

    # 如果从系统获取今天的日期
    if GET_TODAY_FROM_SYSTEM:
        # 从系统获取今天的日期
        month_today, day_today, year_today = get_date_from_system()
    else:
        # 从用户获取今天的日期
        month_today, day_today, year_today = get_date_from_user(
            "ENTER TODAY'S DATE IN THE FORM: 3,24,1979"
        )

    # This program determines the day of the week
    # for a date after 1582

    print()
    # 从用户输入中获取月、日、年
    month, day, year = get_date_from_user(
        "ENTER DAY OF BIRTH (OR OTHER DAY OF INTEREST) (like MM,DD,YYYY)"
    )

    print()

    # 检查日期是否在1582年之前，如果是则打印提示信息并结束程序
    if year < 1582:
        print("NOT PREPARED TO GIVE DAY OF WEEK PRIOR TO MDLXXXII.")
        end()
        return

    # 计算输入日期是星期几
    b = calculate_day_of_week(year, month, day)

    # 计算今天的日期对应的数值
    today_day_value = calc_day_value(year_today, month_today, day_today)
    # 计算目标日期对应的数值
    target_day_value = calc_day_value(year, month, day)

    # 检查目标日期是否在今天之前
    is_today = False

    if today_day_value < target_day_value:
        label = "WILL BE A"  # 设置标签为“将是”
    elif today_day_value == target_day_value:  # 如果今天的日期值等于目标日期值
        label = "IS A"  # 设置标签为“是”
        is_today = True  # 将is_today标记为True
    else:  # 其他情况
        label = "WAS A"  # 设置标签为“曾是”

    day_name = get_day_of_week(b, day)  # 获取日期对应的星期几名称

    # 打印日期所在的星期几
    print(f"{month}/{day}/{year} {label} {day_name}.")

    if is_today:  # 如果是今天
        # 今天没有报告
        end()  # 结束程序
        return  # 返回

    print()  # 打印空行

    el_years = year_today - year  # 计算年份差
    el_months = month_today - month  # 计算当前月份与目标月份的差值
    el_days = day_today - day  # 计算当前日期与目标日期的差值

    if el_days < 0:  # 如果目标日期早于当前日期
        el_months = el_months - 1  # 月份差值减一
        el_days = el_days + 30  # 天数差值加上30（假设每个月30天）
    if el_months < 0:  # 如果目标月份早于当前月份
        el_years = el_years - 1  # 年份差值减一
        el_months = el_months + 12  # 月份差值加上12
    if el_years < 0:  # 如果目标年份早于当前年份
        # target date is in the future  # 目标日期在未来
        end()  # 调用end函数
        return  # 返回

    if (el_months == 0) and (el_days == 0):  # 如果月份差值为0且天数差值为0
        print("***HAPPY BIRTHDAY***")  # 打印生日祝福语

    # print report  # 打印报告
    print(" " * 23 + "\tYEARS\tMONTHS\tDAYS")  # 打印表头
    print(" " * 23 + "\t-----\t------\t----")  # 打印分隔线
    print(f"YOUR AGE (IF BIRTHDATE)\t{el_years}\t{el_months}\t{el_days}")  # 打印出生日期对应的年龄

    life_days = (el_years * 365) + (el_months * 30) + el_days + int(el_months / 2)  # 计算总共活了多少天
    rem_years = el_years  # 剩余年份等于初始年份
    rem_months = el_months  # 剩余月份等于初始月份
    rem_days = el_days  # 剩余天数等于初始天数

    rem_years, rem_months, rem_days, used_years, used_months, used_days = deduct_time(
        0.35, life_days, rem_years, rem_months, rem_days
    )  # 调用deduct_time函数，计算时间的剩余和使用情况
    time_report("YOU HAVE SLEPT", used_years, used_months, used_days)  # 打印出睡眠时间的使用情况
    rem_years, rem_months, rem_days, used_years, used_months, used_days = deduct_time(
        0.17, life_days, rem_years, rem_months, rem_days
    )  # 调用deduct_time函数，计算时间的剩余和使用情况
    time_report("YOU HAVE EATEN", used_years, used_months, used_days)  # 打印出进食时间的使用情况

    label = make_occupation_label(rem_years)  # 调用make_occupation_label函数，生成职业标签
    rem_years, rem_months, rem_days, used_years, used_months, used_days = deduct_time(
        0.23, life_days, rem_years, rem_months, rem_days
    )  # 调用deduct_time函数，计算时间的剩余和使用情况
    time_report("YOU HAVE " + label, used_years, used_months, used_days)  # 调用 time_report 函数，传入参数计算已经使用的年、月、日，并输出结果
    time_report("YOU HAVE RELAXED", rem_years, rem_months, rem_days)  # 调用 time_report 函数，传入参数计算剩余的年、月、日，并输出结果

    print()  # 输出空行

    # 计算退休日期
    e = year + 65  # 计算退休年份
    print(" " * 16 + f"***  YOU MAY RETIRE IN {e} ***")  # 输出可能的退休年份
    end()  # 调用 end 函数结束程序


if __name__ == "__main__":
    main()  # 调用 main 函数，程序入口
```