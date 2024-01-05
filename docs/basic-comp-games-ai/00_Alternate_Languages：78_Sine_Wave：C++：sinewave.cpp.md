# `00_Alternate_Languages\78_Sine_Wave\C++\sinewave.cpp`

```
#include <iostream> // 包含输入输出流库
#include <string>	// 包含字符串库
#include <cmath>	// 包含数学库中的三角函数

int main()
{
	std::cout << std::string(30, ' ') << "SINE WAVE" << std::endl; // 输出一个包含30个空格和"SINE WAVE"的字符串
	std::cout << std::string(15, ' ') << "CREATIVE COMPUTING MORRISTOWN, NEW JERSEY" << std::endl; // 输出一个包含15个空格和"CREATIVE COMPUTING MORRISTOWN, NEW JERSEY"的字符串
	std::cout << std::string(5, '\n'); // 输出包含5个换行符的字符串

	bool b = true; // 声明并初始化布尔变量b为true

	for (double t = 0.0; t <= 40.0; t += 0.25) // 循环，t从0.0开始，每次增加0.25，直到t大于40.0
	{
		int a = int(26 + 25 * std::sin(t)); // 计算a的值为26加上25乘以sin(t)的整数部分
		std::cout << std::string(a, ' ') << (b ? "CREATIVE" : "COMPUTING") << std::endl; // 输出一个包含a个空格和"CREATIVE"或"COMPUTING"的字符串，取决于b的值
		b = !b; // 将b的值取反
	}

	return 0; // 返回0，表示程序正常结束
}
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```