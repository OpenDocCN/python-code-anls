# `basic-computer-games\77_Salvo\csharp\Extensions\RandomExtensions.cs`

```

// 命名空间声明，表示该类属于 Games.Common.Randomness 命名空间
namespace Games.Common.Randomness;

// 声明一个静态类 RandomExtensions
internal static class RandomExtensions
{
    // 声明一个扩展方法 NextShipPosition，用于生成随机的船只位置
    internal static (Position, Offset) NextShipPosition(this IRandom random)
    {
        // 生成随机的起始坐标
        var startX = random.NextCoordinate();
        var startY = random.NextCoordinate();
        // 生成随机的偏移量
        var deltaY = random.NextOffset();
        var deltaX = random.NextOffset();
        // 返回起始坐标和偏移量组成的元组
        return (new(startX, startY), new(deltaX, deltaY));
    }

    // 声明一个扩展方法 NextCoordinate，用于生成随机的坐标
    private static Coordinate NextCoordinate(this IRandom random)
        => random.Next(Coordinate.MinValue, Coordinate.MaxValue + 1);

    // 声明一个私有方法 NextOffset，用于生成随机的偏移量
    private static int NextOffset(this IRandom random) => random.Next(-1, 2);

    // 声明一个扩展方法 GetRandomShipPositionInRange，用于生成在指定范围内的随机船只位置
    internal static (Position, Offset) GetRandomShipPositionInRange(this IRandom random, int shipSize)
    {
        // 循环直到找到合适的船只位置
        while (true)
        {
            // 生成随机的起始坐标和偏移量
            var (start, delta) = random.NextShipPosition();
            var shipSizeLessOne = shipSize - 1;
            // 计算船只的结束坐标
            var end = start + delta * shipSizeLessOne;
            // 如果偏移量不为0且结束坐标在合理范围内，则返回起始坐标和偏移量
            if (delta != 0 && end.IsInRange) 
            {
                return (start, delta);
            }
        }
    }
}

```