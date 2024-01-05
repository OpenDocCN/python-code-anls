# `77_Salvo\csharp\Extensions\RandomExtensions.cs`

```
namespace Games.Common.Randomness;

internal static class RandomExtensions
{
    // 定义一个静态方法，用于生成随机的船只位置
    internal static (Position, Offset) NextShipPosition(this IRandom random)
    {
        // 生成随机的起始 x 坐标
        var startX = random.NextCoordinate();
        // 生成随机的起始 y 坐标
        var startY = random.NextCoordinate();
        // 生成随机的 y 方向偏移量
        var deltaY = random.NextOffset();
        // 生成随机的 x 方向偏移量
        var deltaX = random.NextOffset();
        // 返回一个包含起始位置和偏移量的元组
        return (new Position(startX, startY), new Offset(deltaX, deltaY));
    }

    // 定义一个扩展方法，用于生成随机的坐标值
    private static Coordinate NextCoordinate(this IRandom random)
        => random.Next(Coordinate.MinValue, Coordinate.MaxValue + 1);

    // 定义一个扩展方法，用于生成随机的偏移量
    private static int NextOffset(this IRandom random) => random.Next(-1, 2);

    // 定义一个静态方法，用于在指定范围内生成随机的船只位置
    internal static (Position, Offset) GetRandomShipPositionInRange(this IRandom random, int shipSize)
    {
        while (true)  # 创建一个无限循环
        {
            var (start, delta) = random.NextShipPosition();  # 调用 random 对象的 NextShipPosition 方法，获取起始位置和方向
            var shipSizeLessOne = shipSize - 1;  # 计算船的大小减一
            var end = start + delta * shipSizeLessOne;  # 计算船的结束位置
            if (delta != 0 && end.IsInRange)  # 检查方向不为零且结束位置在范围内
            {
                return (start, delta);  # 返回起始位置和方向
            }
        }
    }
}
```