# `basic-computer-games\00_Alternate_Languages\78_Sine_Wave\C++\sinewave.cpp`

```py
#include <iostream> // 包含输入输出流库
#include <string>    // 包含字符串库，用于创建字符串
#include <cmath>    // 包含数学库，用于计算正弦值

int main()
{
    std::cout << std::string(30, ' ') << "SINE WAVE" << std::endl; // 输出包含30个空格的字符串和"SINE WAVE"
    std::cout << std::string(15, ' ') << "CREATIVE COMPUTING MORRISTOWN, NEW JERSEY" << std::endl; // 输出包含15个空格的字符串和"CREATIVE COMPUTING MORRISTOWN, NEW JERSEY"
    std::cout << std::string(5, '\n'); // 输出包含5个换行符的字符串

    bool b = true; // 声明并初始化布尔变量b为true

    for (double t = 0.0; t <= 40.0; t += 0.25) // 循环，t从0.0开始，每次增加0.25，直到t大于40.0
    {
        int a = int(26 + 25 * std::sin(t)); // 计算sin(t)的值，乘以25再加上26，转换为整数赋值给a
        std::cout << std::string(a, ' ') << (b ? "CREATIVE" : "COMPUTING") << std::endl; // 输出包含a个空格的字符串和"CREATIVE"或"COMPUTING"，取决于b的值
        b = !b; // 取反b的值
    }

    return 0; // 返回0
}
```