# `basic-computer-games\21_Calendar\javascript\calendar.js`

```

// 定义一个打印函数，将字符串添加到输出元素中
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

// 1979年的第一天是星期一
var d = -1;
var s = 0;

// 遍历12个月
for (n = 1; n <= 12; n++) {
	// 计算每个月的天数总和
	s = s + m[n - 1];
	str = "**" + s;
	while (str.length < 7)
		str += " ";
	for (i = 1; i <= 18; i++)
		str += "*";
	// 根据月份添加标题
	switch (n) {
		// ...
	}
	for (i = 1; i <= 18; i++)
		str += "*";
	str += (365 - s) + "**";
	// 打印月份的天数总和
	print(str + "\n");
	// 打印星期标题
	print("     S       M       T       W       T       F       S\n");
	// 打印空行
	print("\n");
	// 打印星期行
	// ...
}

```