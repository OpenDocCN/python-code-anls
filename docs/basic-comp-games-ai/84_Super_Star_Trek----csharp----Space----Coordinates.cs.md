# `basic-computer-games\84_Super_Star_Trek\csharp\Space\Coordinates.cs`

```py
// 使用 System 命名空间
using System;
// 使用 SuperStarTrek.Utils 命名空间
using SuperStarTrek.Utils;

// 表示星系中一个象限的坐标，或者一个象限中的一个区块的坐标
// 注意原点在左上角，x 增加向下，y 增加向右
internal record Coordinates
{
    internal Coordinates(int x, int y)
    {
        // 对 x 和 y 进行验证
        X = Validated(x, nameof(x));
        Y = Validated(y, nameof(y));

        // 计算区域索引
        RegionIndex = (X << 1) + (Y >> 2);
        // 计算子区域索引
        SubRegionIndex = Y % 4;
    }

    internal int X { get; } // x 坐标

    internal int Y { get; } // y 坐标

    internal int RegionIndex { get; } // 区域索引

    internal int SubRegionIndex { get; } // 子区域索引

    // 对值进行验证
    private static int Validated(int value, string argumentName)
    {
        if (value >= 0 && value <= 7) { return value; }

        throw new ArgumentOutOfRangeException(argumentName, value, "Must be 0 to 7 inclusive");
    }

    // 检查值是否有效
    private static bool IsValid(int value) => value >= 0 && value <= 7;

    // 重写 ToString 方法
    public override string ToString() => $"{X+1} , {Y+1}";

    // 解构方法
    internal void Deconstruct(out int x, out int y)
    {
        x = X;
        y = Y;
    }

    // 尝试创建坐标
    internal static bool TryCreate(float x, float y, out Coordinates coordinates)
    {
        var roundedX = Round(x);
        var roundedY = Round(y);

        if (IsValid(roundedX) && IsValid(roundedY))
        {
            coordinates = new Coordinates(roundedX, roundedY);
            return true;
        }

        coordinates = default;
        return false;

        // 四舍五入方法
        static int Round(float value) => (int)Math.Round(value, MidpointRounding.AwayFromZero);
    }

    // 获取到目标坐标的方向和距离
    internal (float Direction, float Distance) GetDirectionAndDistanceTo(Coordinates destination) =>
        DirectionAndDistance.From(this).To(destination);

    // 获取到目标坐标的距离
    internal float GetDistanceTo(Coordinates destination)
    {
        var (_, distance) = GetDirectionAndDistanceTo(destination);
        return distance;
    }
}
```