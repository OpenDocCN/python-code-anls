# `basic-computer-games\21_Calendar\javascript\calendar.js`

```py
// 定义一个打印函数，将字符串添加到指定元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 打印日历标题
print(tab(32) + "CALENDAR\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
print("\n");
print("\n");
print("\n");

// 定义每个月的天数数组
var m = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

// 1979年的第一天是星期一，初始化变量d和s
for (i = 1; i <= 6; i++)
    print("\n");

// 初始化变量d为-1
d = -1;    // 1979 starts on Monday (0 = Sun, -1 = Monday, -2 = Tuesday)
s = 0;

// 遍历12个月
for (n = 1; n <= 12; n++) {
    print("\n");
    print("\n");
    s = s + m[n - 1];
    str = "**" + s;
    while (str.length < 7)
        str += " ";
    for (i = 1; i <= 18; i++)
        str += "*";
    // 根据月份添加对应的月份名称
    switch (n) {
        case  1:    str += " JANUARY "; break;
        case  2:    str += " FEBRUARY"; break;
        case  3:    str += "  MARCH  "; break;
        case  4:    str += "  APRIL  "; break;
        case  5:    str += "   MAY   "; break;
        case  6:    str += "   JUNE  "; break;
        case  7:    str += "   JULY  "; break;
        case  8:    str += "  AUGUST "; break;
        case  9:    str += "SEPTEMBER"; break;
        case 10:    str += " OCTOBER "; break;
        case 11:    str += " NOVEMBER"; break;
        case 12:    str += " DECEMBER"; break;
    }
    for (i = 1; i <= 18; i++)
        str += "*";
    str += (365 - s) + "**";
    // 打印月份标题
    print(str + "\n");
    // 打印星期标题
    print("     S       M       T       W       T       F       S\n");
    print("\n");
    str = "";
    for (i = 1; i <= 59; i++)
        str += "*";
}
    # 循环遍历每周
    for (week = 1; week <= 6; week++) {
        # 打印字符串并换行
        print(str + "\n");
        # 重置字符串
        str = "    ";
        # 循环遍历每天
        for (g = 1; g <= 7; g++) {
            # 递增日期
            d++;
            # 计算日期与月初的差值
            d2 = d - s;
            # 如果日期超过了本月的天数，则跳出循环
            if (d2 > m[n]) {
                week = 6;
                break;
            }
            # 如果日期大于0，则添加到字符串中
            if (d2 > 0)
                str += d2;
            # 补齐字符串长度
            while (str.length < 4 + 8 * g)
                str += " ";
        }
        # 如果日期等于本月的天数，则跳出循环
        if (d2 == m[n]) {
            d += g;
            break;
        }
    }
    # 减去多余的天数
    d -= g;
    # 打印字符串并换行
    print(str + "\n");
# 闭合前面的函数定义
```