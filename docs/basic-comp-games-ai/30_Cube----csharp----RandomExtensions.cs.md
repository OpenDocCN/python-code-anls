# `basic-computer-games\30_Cube\csharp\RandomExtensions.cs`

```py
# 创建一个名为 Cube 的命名空间，并定义一个内部静态类 RandomExtensions
namespace Cube;

internal static class RandomExtensions
{
    # 定义一个扩展方法 NextLocation，接收一个名为 bias 的三元组参数，并返回一个包含三个浮点数的元组
    internal static (float, float, float) NextLocation(this IRandom random, (int, int, int) bias)
        => (random.NextCoordinate(bias.Item1), random.NextCoordinate(bias.Item2), random.NextCoordinate(bias.Item3));

    # 定义一个私有静态方法 NextCoordinate，接收一个名为 bias 的整数参数，并返回一个浮点数
    private static float NextCoordinate(this IRandom random, int bias)
    {
        # 生成一个范围在 0 到 2 之间的随机整数
        var value = random.Next(3);
        # 如果随机整数为 0，则将其替换为 bias
        if (value == 0) { value = bias; }
        # 返回处理后的随机整数
        return value;
    }
}
```