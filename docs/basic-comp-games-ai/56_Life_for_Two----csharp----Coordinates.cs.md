# `basic-computer-games\56_Life_for_Two\csharp\Coordinates.cs`

```py
// 命名空间声明
namespace LifeforTwo;

// 声明一个内部的记录类型 Coordinates，包含 X 和 Y 两个整型字段
internal record Coordinates (int X, int Y)
{
    // 声明一个静态方法，用于尝试创建 Coordinates 对象
    public static bool TryCreate((float X, float Y) values, out Coordinates coordinates)
    {
        // 如果传入的值不在指定范围内，则返回默认坐标 (0, 0) 并返回 false
        if (values.X <= 0 || values.X > 5 || values.Y <= 0 || values.Y > 5)
        {
            coordinates = new(0, 0);
            return false;
        }

        // 根据传入的值创建 Coordinates 对象，并返回 true
        coordinates = new((int)values.X, (int)values.Y);
        return true;
    }

    // 声明一个重载运算符 +，用于对 Coordinates 对象进行加法运算
    public static Coordinates operator +(Coordinates coordinates, int value) =>
        new (coordinates.X + value, coordinates.Y + value);

    // 声明一个方法，用于获取当前坐标的邻居坐标
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