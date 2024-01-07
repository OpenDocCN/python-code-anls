# `basic-computer-games\25_Chief\csharp\Math.cs`

```

# 引入 Chief.Resources.Resource 命名空间中的静态资源
using static Chief.Resources.Resource;

# 定义名为 Math 的静态类
namespace Chief;

public static class Math
{
    # 定义名为 CalculateOriginal 的静态方法，用于计算原始值
    public static float CalculateOriginal(float result) => (result + 1 - 5) * 5 / 8 * 5 - 3;

    # 定义名为 ShowWorking 的静态方法，用于展示计算过程
    public static string ShowWorking(Number value) =>
        # 格式化输出计算过程
        string.Format(
            Formats.Working,  # 使用 Formats.Working 格式
            value,             # 原始值
            value += 3,        # 加 3
            value /= 5,        # 除以 5
            value *= 8,        # 乘以 8
            value = value / 5 + 5,  # 除以 5 再加 5
            value - 1);        # 减 1
}

```