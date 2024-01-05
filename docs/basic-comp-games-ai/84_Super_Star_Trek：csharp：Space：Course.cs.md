# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\Space\Course.cs`

```
// 使用System和System.Collections.Generic命名空间
using System;
using System.Collections.Generic;

namespace SuperStarTrek.Space
{
    // 实现了原始代码中的航向计算
    //     530 FORI=1TO9:C(I,1)=0:C(I,2)=0:NEXTI
    //     540 C(3,1)=-1:C(2,1)=-1:C(4,1)=-1:C(4,2)=-1:C(5,2)=-1:C(6,2)=-1
    //     600 C(1,2)=1:C(2,2)=1:C(6,1)=1:C(7,1)=1:C(8,1)=1:C(8,2)=1:C(9,2)=1
    internal class Course
    {
        // 创建一个只读的元组数组，表示方向的变化
        private static readonly (int DeltaX, int DeltaY)[] cardinals = new[]
        {
            (0, 1),  // 北
            (-1, 1), // 西北
            (-1, 0), // 西
            (-1, -1), // 西南
        (0, -1),   # 定义一个元组，表示向上移动的偏移量
        (1, -1),   # 定义一个元组，表示向右上移动的偏移量
        (1, 0),    # 定义一个元组，表示向右移动的偏移量
        (1, 1),    # 定义一个元组，表示向右下移动的偏移量
        (0, 1)     # 定义一个元组，表示向下移动的偏移量
    };

    internal Course(float direction)   # 定义一个内部的 Course 类，接受一个浮点数参数 direction
    {
        if (direction < 1 || direction > 9)   # 如果 direction 不在 1 到 9 的范围内
        {
            throw new ArgumentOutOfRangeException(   # 抛出参数超出范围的异常
                nameof(direction),   # 异常参数的名称
                direction,   # 异常参数的值
                "Must be between 1 and 9, inclusive.");   # 异常的描述信息
        }

        var cardinalDirection = (int)(direction - 1) % 8;   # 计算方向的基本方向
        var fractionalDirection = direction - (int)direction;   # 计算方向的小数部分
        // 获取当前基本方向的坐标
        var baseCardinal = cardinals[cardinalDirection];
        // 获取下一个基本方向的坐标
        var nextCardinal = cardinals[cardinalDirection + 1];

        // 根据当前方向和下一个方向的坐标差值，计算 X 轴的增量
        DeltaX = baseCardinal.DeltaX + (nextCardinal.DeltaX - baseCardinal.DeltaX) * fractionalDirection;
        // 根据当前方向和下一个方向的坐标差值，计算 Y 轴的增量
        DeltaY = baseCardinal.DeltaY + (nextCardinal.DeltaY - baseCardinal.DeltaY) * fractionalDirection;
    }

    // 获取 X 轴的增量
    internal float DeltaX { get; }
    // 获取 Y 轴的增量
    internal float DeltaY { get; }

    // 从起始坐标开始，根据增量获取坐标点
    internal IEnumerable<Coordinates> GetSectorsFrom(Coordinates start)
    {
        (float x, float y) = start;

        // 循环计算下一个坐标点
        while(true)
        {
            x += DeltaX;
            y += DeltaY;
            if (!Coordinates.TryCreate(x, y, out var coordinates))
            {
                yield break;  # 如果无法创建坐标对象，则结束当前循环
            }

            yield return coordinates;  # 返回坐标对象
        }
    }

    internal (bool, Coordinates, Coordinates) GetDestination(Coordinates quadrant, Coordinates sector, int distance)
    {
        var (xComplete, quadrantX, sectorX) = GetNewCoordinate(quadrant.X, sector.X, DeltaX * distance);  # 获取新的 X 坐标
        var (yComplete, quadrantY, sectorY) = GetNewCoordinate(quadrant.Y, sector.Y, DeltaY * distance);  # 获取新的 Y 坐标

        return (xComplete && yComplete, new Coordinates(quadrantX, quadrantY), new Coordinates(sectorX, sectorY));  # 返回是否成功以及新的坐标对象
    }

    private static (bool, int, int) GetNewCoordinate(int quadrant, int sector, float sectorsTravelled)
    {
        var galacticCoordinate = quadrant * 8 + sector + sectorsTravelled;  # 计算新的坐标值
        # 计算新的象限
        var newQuadrant = (int)(galacticCoordinate / 8);
        # 计算新的扇区
        var newSector = (int)(galacticCoordinate - newQuadrant * 8);

        # 如果新的扇区小于0，则调整新的象限和扇区
        if (newSector < 0)
        {
            newQuadrant -= 1;
            newSector += 8;
        }

        # 根据新的象限进行判断，返回对应的结果元组
        return newQuadrant switch
        {
            # 如果新的象限小于0，则返回(假, 0, 0)
            < 0 => (false, 0, 0),
            # 如果新的象限大于7，则返回(假, 7, 7)
            > 7 => (false, 7, 7),
            # 否则返回(真, 新的象限, 新的扇区)
            _ => (true, newQuadrant, newSector)
        };
    }
}
```