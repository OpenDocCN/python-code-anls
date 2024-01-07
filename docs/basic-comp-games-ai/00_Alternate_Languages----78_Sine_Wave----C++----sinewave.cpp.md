# `basic-computer-games\00_Alternate_Languages\78_Sine_Wave\C++\sinewave.cpp`

```

#include <iostream> // 包含输入输出流库，用于输出信息到控制台
#include <string>	// 包含字符串库，用于创建和操作字符串
#include <cmath>	// 包含数学库，用于计算正弦值

int main()
{
	// 输出标题
	std::cout << std::string(30, ' ') << "SINE WAVE" << std::endl;
	// 输出副标题
	std::cout << std::string(15, ' ') << "CREATIVE COMPUTING MORRISTOWN, NEW JERSEY" << std::endl;
	// 输出空行
	std::cout << std::string(5, '\n');

	// 初始化布尔变量
	bool b = true;

	// 循环计算正弦波形
	for (double t = 0.0; t <= 40.0; t += 0.25)
	{
		// 计算波形高度
		int a = int(26 + 25 * std::sin(t));
		// 输出波形
		std::cout << std::string(a, ' ') << (b ? "CREATIVE" : "COMPUTING") << std::endl;
		// 切换输出内容
		b = !b;
	}

	return 0;
}

```