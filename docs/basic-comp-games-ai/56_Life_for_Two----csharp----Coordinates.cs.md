# `basic-computer-games\56_Life_for_Two\csharp\Coordinates.cs`

```

namespace LifeforTwo;

// 定义一个内部的记录类型 Coordinates，包含 X 和 Y 两个整型字段
internal record Coordinates (int X, int Y)
{
    // 定义一个静态方法，用于创建 Coordinates 对象，如果传入的值不在指定范围内，则返回 false
    public static bool TryCreate((float X, float Y) values, out Coordinates coordinates)
    {
        // 如果传入的值不在指定范围内，则将 coordinates 设置为 (0, 0)，并返回 false
        if (values.X <= 0 || values.X > 5 || values.Y <= 0 || values.Y > 5)
        {
            coordinates = new(0, 0);
            return false;
        }

        // 将传入的值转换为整型，并赋给 coordinates
        coordinates = new((int)values.X, (int)values.Y);
        return true;
    }

    // 定义一个重载运算符 +，用于实现 Coordinates 对象与整数相加
    public static Coordinates operator +(Coordinates coordinates, int value) =>
        new (coordinates.X + value, coordinates.Y + value);

    // 定义一个方法，用于获取当前坐标的相邻坐标
    public IEnumerable<Coordinates> GetNeighbors()
    {
        yield return new(X - 1, Y);
        yield return new(X + 1, Y);
        yield return new(X, Y - 1);
        yield return new(X, Y + 1);
        yield return new(X - 1, Y - 1);
        yield return new(X + 1, Y - 1);
        yield return new(X - 1, Y + 1);
        yield return new(X + 1, Y + 1);
    }
}

```