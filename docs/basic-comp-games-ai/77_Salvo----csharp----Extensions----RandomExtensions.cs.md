# `basic-computer-games\77_Salvo\csharp\Extensions\RandomExtensions.cs`

```
# 命名空间 Games.Common.Randomness 下的内部静态类 RandomExtensions
internal static class RandomExtensions
{
    # 返回一个随机的船只位置和偏移量
    internal static (Position, Offset) NextShipPosition(this IRandom random)
    {
        # 获取随机的起始 x 坐标
        var startX = random.NextCoordinate();
        # 获取随机的起始 y 坐标
        var startY = random.NextCoordinate();
        # 获取随机的 y 轴偏移量
        var deltaY = random.NextOffset();
        # 获取随机的 x 轴偏移量
        var deltaX = random.NextOffset();
        # 返回起始位置和偏移量的元组
        return (new(startX, startY), new(deltaX, deltaY));
    }

    # 返回一个随机的坐标
    private static Coordinate NextCoordinate(this IRandom random)
        => random.Next(Coordinate.MinValue, Coordinate.MaxValue + 1);

    # 返回一个随机的偏移量
    private static int NextOffset(this IRandom random) => random.Next(-1, 2);

    # 返回一个在指定船只大小范围内的随机船只位置和偏移量
    internal static (Position, Offset) GetRandomShipPositionInRange(this IRandom random, int shipSize)
    {
        # 无限循环，直到找到合适的船只位置和偏移量
        while (true)
        {
            # 获取随机的起始位置和偏移量
            var (start, delta) = random.NextShipPosition();
            # 计算船只的大小减一
            var shipSizeLessOne = shipSize - 1;
            # 计算船只的结束位置
            var end = start + delta * shipSizeLessOne;
            # 如果偏移量不为零且结束位置在范围内，则返回起始位置和偏移量
            if (delta != 0 && end.IsInRange) 
            {
                return (start, delta);
            }
        }
    }
}
```