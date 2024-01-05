# `d:/src/tocomm/basic-computer-games\21_Calendar\javascript\calendar.js`

```
// CALENDAR
// 该程序是一个日历程序，将BASIC语言转换为Javascript语言，作者是Oscar Toledo G. (nanochess)

// 定义一个打印函数，将字符串添加到指定id的元素中
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
// 打印程序信息
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
// 打印换行
print("\n");
# 打印两个换行符
print("\n");
print("\n");

# 定义一个数组，存储每个月份的天数
var m = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

# 1979年的值 - 参见注释
for (i = 1; i <= 6; i++)
    print("\n");

# 初始化变量d为-1，表示1979年的第一天是星期一（0 = 星期日，-1 = 星期一，-2 = 星期二）
d = -1;
s = 0;

# 遍历12个月份
for (n = 1; n <= 12; n++) {
    print("\n");
    print("\n");
    # 计算到当前月份为止的总天数
    s = s + m[n - 1];
    str = "**" + s;
    # 如果字符串长度小于7，则在末尾添加空格
    while (str.length < 7)
        str += " ";
# 使用循环将字符串"*"添加18次到变量str中
for (i = 1; i <= 18; i++)
    str += "*";

# 使用switch语句根据变量n的值将对应的月份名称添加到str中
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

# 使用循环将字符串"*"再次添加18次到变量str中
for (i = 1; i <= 18; i++)
    str += "*";

# 根据条件将365 - s或366 - s（在闰年）添加到str中
str += (365 - s) + "**";
// 366 - s on leap years
	# 打印字符串并换行
	print(str + "\n");
	# 打印星期标题
	print("     S       M       T       W       T       F       S\n");
	# 打印换行
	print("\n");
	# 重置字符串
	str = "";
	# 循环59次，向字符串中添加"*"
	for (i = 1; i <= 59; i++)
		str += "*";
	# 循环6次，表示6个星期
	for (week = 1; week <= 6; week++) {
		# 打印字符串并换行
		print(str + "\n");
		# 重置字符串
		str = "    ";
		# 循环7次，表示一周7天
		for (g = 1; g <= 7; g++) {
			# 日期递增
			d++;
			# 计算日期与起始日期的差值
			d2 = d - s;
			# 如果差值大于月份的天数，则跳出循环
			if (d2 > m[n]) {
				week = 6;
				break;
			}
			# 如果差值大于0，则添加到字符串中
			if (d2 > 0)
				str += d2;
			# 补充空格，使得每个日期占据4个字符的位置
			while (str.length < 4 + 8 * g)
				str += " ";
		}  # 结束 if 语句块
		if (d2 == m[n]) {  # 如果 d2 等于 m[n]
			d += g;  # 将 d 增加 g
			break;  # 跳出循环
		}
	}  # 结束 for 循环
	d -= g;  # 将 d 减去 g
	print(str + "\n");  # 打印 str，并换行
}  # 结束函数
```