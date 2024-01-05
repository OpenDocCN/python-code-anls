# `d:/src/tocomm/basic-computer-games\30_Cube\csharp\RandomExtensions.cs`

```
namespace Cube;  # 命名空间声明，定义了代码所在的命名空间

internal static class RandomExtensions  # 定义了一个静态类 RandomExtensions，用于扩展 Random 类的功能

{
    internal static (float, float, float) NextLocation(this IRandom random, (int, int, int) bias)
        => (random.NextCoordinate(bias.Item1), random.NextCoordinate(bias.Item2), random.NextCoordinate(bias.Item3));
    # 定义了一个扩展方法 NextLocation，接受一个 IRandom 类型的参数和一个三元组 bias，返回一个三元组 (float, float, float)

    private static float NextCoordinate(this IRandom random, int bias)
    {
        var value = random.Next(3);  # 调用 IRandom 接口的 Next 方法，返回一个 0 到 2 之间的随机整数
        if (value == 0) { value = bias; }  # 如果随机数为 0，则将其替换为 bias
        return value;  # 返回随机数或者 bias
    }
}
```