# `basic-computer-games\72_Queen\csharp\RandomExtensions.cs`

```
# 命名空间 Queen 下的内部静态类 RandomExtensions
namespace Queen;

# 内部静态类 RandomExtensions 中的扩展方法，用于生成下一步棋子的移动
internal static class RandomExtensions
{
    # 扩展方法，用于生成下一步棋子的移动
    internal static Move NextMove(this IRandom random)
        # 调用 IRandom 接口的 NextFloat 方法，根据返回值进行匹配
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