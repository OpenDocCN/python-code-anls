# `d:/src/tocomm/basic-computer-games\25_Chief\csharp\Math.cs`

```
using static Chief.Resources.Resource;  // 导入 Chief.Resources.Resource 命名空间中的所有静态成员

namespace Chief;  // 定义 Chief 命名空间

public static class Math  // 定义名为 Math 的静态类
{
    public static float CalculateOriginal(float result) => (result + 1 - 5) * 5 / 8 * 5 - 3;  // 定义名为 CalculateOriginal 的静态方法，用于计算原始值

    public static string ShowWorking(Number value) =>  // 定义名为 ShowWorking 的静态方法，用于展示计算过程
        string.Format(
            Formats.Working,  // 使用 Formats.Working 格式化字符串
            value,  // 将 value 作为参数传入格式化字符串
            value += 3,  // 将 value 加 3，并将结果作为参数传入格式化字符串
            value /= 5,  // 将 value 除以 5，并将结果作为参数传入格式化字符串
            value *= 8,  // 将 value 乘以 8，并将结果作为参数传入格式化字符串
            value = value / 5 + 5,  // 将 value 除以 5，加上 5，并将结果作为参数传入格式化字符串
            value - 1);  // 将 value 减 1，并将结果作为参数传入格式化字符串
}
```