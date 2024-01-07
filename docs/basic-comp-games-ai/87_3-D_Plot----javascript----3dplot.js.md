# `basic-computer-games\87_3-D_Plot\javascript\3dplot.js`

```

// 3D PLOT
//
// 由Oscar Toledo G. (nanochess)将BASIC转换为Javascript
//
// 打印函数，将字符串添加到输出元素中
function print(str)
{
	document.getElementById("output").appendChild(document.createTextNode(str));
}

// 制表函数，返回指定数量的空格字符串
function tab(space)
{
	var str = "";
	while (space-- > 0)
		str += " ";
	return str;
}

// 方程函数，根据输入计算值并返回结果
function equation(input)
{
	return 30 * Math.exp(-input * input / 100);
}

// 打印标题
print(tab(32) + "3D PLOT\n");
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");

// 循环计算并打印3D图形
for (x = -30; x <= 30; x += 1.5) {
	l = 0;
	y1 = 5 * Math.floor(Math.sqrt(900 - x * x) / 5);
	str = "";
	for (y = y1; y >= -y1; y -= 5) {
		z = Math.floor(25 + equation(Math.sqrt(x * x + y * y)) - .7 * y);
		if (z > l) {
			l = z;
			while (str.length < z)
				str += " ";
			str += "*";
		}
	}
	print(str + "\n");
}

```