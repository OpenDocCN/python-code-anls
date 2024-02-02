# `basic-computer-games\84_Super_Star_Trek\csharp\Space\Course.cs`

```py
// 命名空间 SuperStarTrek.Space 中的 Course 类，实现了原始代码中的航向计算
internal class Course
{
    // 定义一个只读的数组，包含了8个方向的坐标变化
    private static readonly (int DeltaX, int DeltaY)[] cardinals = new[]
    {
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1)
    };

    // 构造函数，接受一个方向参数
    internal Course(float direction)
    {
        // 如果方向不在1到9之间，抛出参数越界异常
        if (direction < 1 || direction > 9)
        {
            throw new ArgumentOutOfRangeException(
                nameof(direction),
                direction,
                "Must be between 1 and 9, inclusive.");
        }

        // 计算基础方向和下一个方向的坐标变化
        var cardinalDirection = (int)(direction - 1) % 8;
        var fractionalDirection = direction - (int)direction;

        var baseCardinal = cardinals[cardinalDirection];
        var nextCardinal = cardinals[cardinalDirection + 1];

        // 计算 DeltaX 和 DeltaY
        DeltaX = baseCardinal.DeltaX + (nextCardinal.DeltaX - baseCardinal.DeltaX) * fractionalDirection;
        DeltaY = baseCardinal.DeltaY + (nextCardinal.DeltaY - baseCardinal.DeltaY) * fractionalDirection;
    }

    // 获取 DeltaX 属性
    internal float DeltaX { get; }

    // 获取 DeltaY 属性
    internal float DeltaY { get; }

    // 从起始坐标开始，根据 DeltaX 和 DeltaY 计算下一个坐标
    internal IEnumerable<Coordinates> GetSectorsFrom(Coordinates start)
    {
        (float x, float y) = start;

        while(true)
        {
            x += DeltaX;
            y += DeltaY;

            // 如果无法创建坐标对象，则结束循环
            if (!Coordinates.TryCreate(x, y, out var coordinates))
            {
                yield break;
            }

            // 返回计算出的坐标
            yield return coordinates;
        }
    }
}
    # 获取目的地坐标
    internal (bool, Coordinates, Coordinates) GetDestination(Coordinates quadrant, Coordinates sector, int distance)
    {
        # 根据当前象限、扇区和距离计算新的 X 坐标
        var (xComplete, quadrantX, sectorX) = GetNewCoordinate(quadrant.X, sector.X, DeltaX * distance);
        # 根据当前象限、扇区和距离计算新的 Y 坐标
        var (yComplete, quadrantY, sectorY) = GetNewCoordinate(quadrant.Y, sector.Y, DeltaY * distance);

        # 返回新的坐标信息
        return (xComplete && yComplete, new Coordinates(quadrantX, quadrantY), new Coordinates(sectorX, sectorY));
    }

    # 计算新的坐标
    private static (bool, int, int) GetNewCoordinate(int quadrant, int sector, float sectorsTravelled)
    {
        # 计算新的星系坐标
        var galacticCoordinate = quadrant * 8 + sector + sectorsTravelled;
        # 计算新的象限
        var newQuadrant = (int)(galacticCoordinate / 8);
        # 计算新的扇区
        var newSector = (int)(galacticCoordinate - newQuadrant * 8);

        # 处理新的扇区小于 0 的情况
        if (newSector < 0)
        {
            newQuadrant -= 1;
            newSector += 8;
        }

        # 根据新的象限判断是否超出范围，返回对应的结果
        return newQuadrant switch
        {
            < 0 => (false, 0, 0),
            > 7 => (false, 7, 7),
            _ => (true, newQuadrant, newSector)
        };
    }
# 闭合前面的函数定义
```