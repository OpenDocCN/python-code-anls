# `21_Calendar\python\calendar.py`

```
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从给定的文件名读取二进制数据，并将其封装成字节流对象
    使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面的内容创建一个 ZIP 对象，以只读模式打开
    遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象中的文件名列表，读取每个文件的数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回包含文件名到数据的字典
from typing import Tuple  # 导入类型提示模块，用于声明函数返回类型

def parse_input() -> Tuple[int, bool]:  # 声明函数的返回类型为元组，包含一个整数和一个布尔值
    """
    function to parse input for weekday and leap year boolean
    """

    days_mapping = {  # 创建一个字典，将星期几映射为对应的整数
        "sunday": 0,
        "monday": -1,
        "tuesday": -2,
        "wednesday": -3,
        "thursday": -4,
        "friday": -5,
        "saturday": -6,
    }

    day = 0  # 初始化变量day为0
    leap_day = False  # 初始化变量 leap_day 为 False，用于记录是否是闰年

    correct_day_input = False  # 初始化变量 correct_day_input 为 False，用于记录是否输入了正确的星期几
    while not correct_day_input:  # 循环直到输入了正确的星期几
        weekday = input("INSERT THE STARTING DAY OF THE WEEK OF THE YEAR:")  # 提示用户输入一周的起始日

        for day_k in days_mapping.keys():  # 遍历 days_mapping 字典的键
            if weekday.lower() in day_k:  # 如果用户输入的星期几在键中
                day = days_mapping[day_k]  # 获取对应的值，即一周的起始日
                correct_day_input = True  # 设置 correct_day_input 为 True，表示输入了正确的星期几
                break  # 跳出循环

    while True:  # 无限循环
        leap = input("IS IT A LEAP YEAR?:")  # 提示用户输入是否是闰年

        if "y" in leap.lower():  # 如果用户输入包含 "y"
            leap_day = True  # 设置 leap_day 为 True，表示是闰年
            break  # 跳出循环

        if "n" in leap.lower():  # 如果用户输入包含 "n"
            leap_day = False  # 初始化leap_day变量为False
            break  # 跳出循环

    return day, leap_day  # 返回day和leap_day变量的值


def calendar(weekday: int, leap_year: bool) -> None:
    """
    function to print a year's calendar.

    input:
        _weekday_: int - the initial day of the week (0=SUN, -1=MON, -2=TUES...)
        _leap_year_: bool - indicates if the year is a leap year
    """
    months_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # 初始化每个月的天数
    days = "S        M        T        W        T        F        S\n"  # 初始化星期的显示格式
    sep = "*" * 59  # 初始化分隔线
    years_day = 365  # 初始化一年的天数
    d = weekday  # 初始化星期的初始值
    if leap_year:  # 如果是闰年
        months_days[2] = 29  # 将二月的天数修改为29
        years_day = 366  # 将年份的天数修改为366

    months_names = [  # 创建包含月份名称的列表
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

    days_count = 0  # 初始化一个变量用于计算天数，原程序中标记为S
    # 主循环
    for n in range(1, 13):
        # 累加每个月的天数
        days_count += months_days[n - 1]
        # 打印当前月份的信息
        print(
            f"** {days_count} ****************** {months_names[n - 1]} "
            f"****************** {years_day - days_count} **\n"
        )
        # 打印天数
        print(days)
        # 打印分隔符
        print(sep)

        # 循环处理每个月的天数
        for _ in range(1, 7):
            print("\n")
            # 循环处理每周的天数
            for g in range(1, 8):  # noqa
                d += 1
                d2 = d - days_count

                # 如果当前天数超过了当前月份的天数，则跳出内层循环
                if d2 > months_days[n]:
                    break
                if d2 <= 0:  # 如果日期小于等于0
                    print("  ", end="       ")  # 打印两个空格
                elif d2 < 10:  # 如果日期小于10
                    print(f" {d2}", end="       ")  # 打印日期并以8个空格结束
                else:  # 否则
                    print(f"{d2}", end="       ")  # 打印日期并以8个空格结束
            print()  # 打印换行

            if d2 >= months_days[n]:  # 如果日期大于等于该月的天数
                break  # 跳出循环

        if d2 > months_days[n]:  # 如果日期大于该月的天数
            d -= g  # 减去g

        print("\n")  # 打印两个换行

    print("\n")  # 打印两个换行


def main() -> None:  # 主函数声明
    print(" " * 32 + "CALENDAR")  # 打印日历标题
    print(" " * 15 + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY")  # 打印创意计算公司信息
    print("\n" * 11)  # 打印11个换行符，用于清空屏幕

    day, leap_year = parse_input()  # 调用parse_input函数，获取用户输入的第一天和是否是闰年
    calendar(day, leap_year)  # 调用calendar函数，传入用户输入的第一天和是否是闰年


if __name__ == "__main__":  # 如果当前脚本被直接执行，而不是被导入
    main()  # 调用main函数

########################################################
#
# Porting notes:
#
# It has been added an input at the beginning of the
# program so the user can specify the first day of the
# week of the year and if the year is leap or not.
# 在程序开头添加了一个输入，以便用户可以指定一年中的第一天是星期几，以及该年是否是闰年。
#
########################################################
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```