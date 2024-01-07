# `basic-computer-games\84_Super_Star_Trek\csharp\Space\Course.cs`

```

// 命名空间声明，定义了类的命名空间
using System;
using System.Collections.Generic;

namespace SuperStarTrek.Space
{
    // 实现了原始代码中的航向计算
    internal class Course
    {
        // 定义了八个基本方向的偏移量
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

        // 构造函数，接受方向参数并计算出 DeltaX 和 DeltaY
        internal Course(float direction)
        {
            // 检查方向参数是否在 1 到 9 之间，否则抛出异常
            if (direction < 1 || direction > 9)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(direction),
                    direction,
                    "Must be between 1 and 9, inclusive.");
            }

            // 计算基本方向和下一个方向的偏移量，并根据方向参数计算出 DeltaX 和 DeltaY
        }

        // DeltaX 属性，表示 X 轴的偏移量
        internal float DeltaX { get; }

        // DeltaY 属性，表示 Y 轴的偏移量
        internal float DeltaY { get; }

        // 从起始坐标开始，根据 DeltaX 和 DeltaY 生成一系列坐标
        internal IEnumerable<Coordinates> GetSectorsFrom(Coordinates start)
        {
            // 循环生成坐标，直到超出范围
        }

        // 根据当前象限、扇区和距离计算目的地的坐标
        internal (bool, Coordinates, Coordinates) GetDestination(Coordinates quadrant, Coordinates sector, int distance)
        {
            // 根据当前坐标和距离计算新的坐标
        }

        // 根据当前象限、扇区和移动的扇区数计算新的坐标
        private static (bool, int, int) GetNewCoordinate(int quadrant, int sector, float sectorsTravelled)
        {
            // 计算新的象限和扇区
        }
    }
}

```