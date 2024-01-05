# `72_Queen\csharp\RandomExtensions.cs`

```
namespace Queen;  # 命名空间声明，定义了代码所在的命名空间

internal static class RandomExtensions  # 定义了一个静态类 RandomExtensions，该类是内部的，只能在当前程序集中访问

{
    internal static Move NextMove(this IRandom random)  # 定义了一个静态方法 NextMove，该方法是内部的，接受一个类型为 IRandom 的参数，并返回一个 Move 类型的值
        => random.NextFloat() switch  # 使用 switch 语句对 random.NextFloat() 的值进行判断
        {
            > 0.6F => Move.Down,  # 如果 random.NextFloat() 的值大于 0.6F，则返回 Move.Down
            > 0.3F => Move.DownLeft,  # 如果 random.NextFloat() 的值大于 0.3F，则返回 Move.DownLeft
            _ => Move.Left  # 其他情况下返回 Move.Left
        };
}
```