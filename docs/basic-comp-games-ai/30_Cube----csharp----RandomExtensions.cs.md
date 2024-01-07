# `basic-computer-games\30_Cube\csharp\RandomExtensions.cs`

```

// 命名空间 Cube 下的内部静态类 RandomExtensions
namespace Cube;

internal static class RandomExtensions
{
    // 扩展方法，用于生成下一个位置坐标
    internal static (float, float, float) NextLocation(this IRandom random, (int, int, int) bias)
        => (random.NextCoordinate(bias.Item1), random.NextCoordinate(bias.Item2), random.NextCoordinate(bias.Item3));

    // 私有静态方法，用于生成下一个坐标值
    private static float NextCoordinate(this IRandom random, int bias)
    {
        // 生成一个 0 到 2 的随机数
        var value = random.Next(3);
        // 如果随机数为 0，则使用偏移值
        if (value == 0) { value = bias; }
        // 返回生成的坐标值
        return value;
    }
}

```