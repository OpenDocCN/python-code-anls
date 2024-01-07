# `basic-computer-games\72_Queen\csharp\RandomExtensions.cs`

```

# 命名空间 Queen 下的内部静态类 RandomExtensions
namespace Queen;

internal static class RandomExtensions
{
    # 扩展方法，用于生成下一步棋的移动
    internal static Move NextMove(this IRandom random)
        # 调用随机数生成器的 NextFloat 方法，根据返回值进行判断
        => random.NextFloat() switch
        {
            # 如果返回值大于 0.6，则返回 Move.Down
            > 0.6F => Move.Down,
            # 如果返回值大于 0.3，则返回 Move.DownLeft
            > 0.3F => Move.DownLeft,
            # 其他情况返回 Move.Left
            _ => Move.Left
        };
}

```