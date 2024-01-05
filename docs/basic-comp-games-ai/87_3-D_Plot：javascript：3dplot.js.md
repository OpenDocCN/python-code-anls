# `d:/src/tocomm/basic-computer-games\87_3-D_Plot\javascript\3dplot.js`

```
// 3D PLOT
// 3D绘图

// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
// 由Oscar Toledo G.（nanochess）将BASIC转换为Javascript

function print(str)
{
	document.getElementById("output").appendChild(document.createTextNode(str));
}
// 定义一个名为print的函数，用于向id为"output"的元素中添加文本节点

function tab(space)
{
	var str = "";
	while (space-- > 0)
		str += " ";
	return str;
}
// 定义一个名为tab的函数，用于返回指定数量的空格字符串

function equation(input)
{
	return 30 * Math.exp(-input * input / 100);
}
// 定义一个名为equation的函数，用于计算给定输入的函数值
}

# 打印3D图形
print(tab(32) + "3D PLOT\n");
# 打印创意计算公司的地址
print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");

# 循环计算3D图形的坐标
for (x = -30; x <= 30; x += 1.5) {
    l = 0;
    y1 = 5 * Math.floor(Math.sqrt(900 - x * x) / 5);
    str = "";
    # 计算每个坐标点的高度
    for (y = y1; y >= -y1; y -= 5) {
        z = Math.floor(25 + equation(Math.sqrt(x * x + y * y)) - .7 * y);
        # 根据高度打印相应数量的空格和星号
        if (z > l) {
            l = z;
            while (str.length < z)
                str += " ";
            str += "*";
        }
    }
    # 打印每一行的图形
    print(str + "\n");
}
bio = BytesIO(open(fname, 'rb').read())  # 根据 ZIP 文件名读取其二进制，封装成字节流
zip = zipfile.ZipFile(bio, 'r')  # 使用字节流里面内容创建 ZIP 对象
fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
zip.close()  # 关闭 ZIP 对象
```