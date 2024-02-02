# `basic-computer-games\25_Chief\csharp\Math.cs`

```py
# 引入 Chief.Resources.Resource 中的静态资源
using static Chief.Resources.Resource;

# 命名空间 Chief
namespace Chief;

# 定义静态类 Math
public static class Math
{
    # 计算原始值的方法
    public static float CalculateOriginal(float result) => (result + 1 - 5) * 5 / 8 * 5 - 3;

    # 展示计算过程的方法
    public static string ShowWorking(Number value) =>
        # 格式化输出计算过程
        string.Format(
            Formats.Working,  # 使用 Formats.Working 格式
            value,  # 原始值
            value += 3,  # 加 3
            value /= 5,  # 除以 5
            value *= 8,  # 乘以 8
            value = value / 5 + 5,  # 除以 5 再加 5
            value - 1);  # 减 1
}
```