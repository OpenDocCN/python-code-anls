# `77_Salvo\csharp\Position.cs`

```
namespace Salvo;  // 命名空间声明

internal record struct Position(Coordinate X, Coordinate Y)  // 定义名为 Position 的记录结构，包含 X 和 Y 两个坐标属性
{
    public bool IsInRange => X.IsInRange && Y.IsInRange;  // 定义 IsInRange 属性，判断 X 和 Y 坐标是否都在范围内
    public bool IsOnDiagonal => X == Y;  // 定义 IsOnDiagonal 属性，判断 X 和 Y 坐标是否在对角线上

    public static Position Create((float X, float Y) coordinates) => new(coordinates.X, coordinates.Y);  // 定义静态方法 Create，根据给定的坐标创建 Position 对象

    public static bool TryCreateValid((float X, float Y) coordinates, out Position position)  // 定义静态方法 TryCreateValid，尝试根据给定的坐标创建有效的 Position 对象
    {
        if (Coordinate.TryCreateValid(coordinates.X, out var x) && Coordinate.TryCreateValid(coordinates.Y, out var y))  // 如果 X 和 Y 坐标都能成功创建有效的 Coordinate 对象
        {
            position = new(x, y);  // 创建新的 Position 对象
            return true;  // 返回 true，表示创建成功
        }

        position = default;  // 否则将 position 设置为默认值
        return false;  // 返回 false，表示创建失败
    }
    public static IEnumerable<Position> All
        => Coordinate.Range.SelectMany(x => Coordinate.Range.Select(y => new Position(x, y)));
    # 返回一个包含所有可能位置的可枚举集合

    public IEnumerable<Position> Neighbours
    {
        get
        {
            foreach (var offset in Offset.Units)
            {
                var neighbour = this + offset;
                if (neighbour.IsInRange) { yield return neighbour; }
            }
        }
    }
    # 返回当前位置的所有邻居位置的可枚举集合

    internal float DistanceTo(Position other)
    {
        var (deltaX, deltaY) = (X - other.X, Y - other.Y);
        return (float)Math.Sqrt(deltaX * deltaX + deltaY * deltaY);
    }
    # 返回当前位置到另一个位置的距离
    }  # 结束 BringIntoRange 方法的定义

    internal Position BringIntoRange(IRandom random)
        => IsInRange ? this : new(X.BringIntoRange(random), Y.BringIntoRange(random));  # 如果当前位置在范围内，则返回当前位置，否则返回一个在范围内的新位置

    public static Position operator +(Position position, Offset offset) 
        => new(position.X + offset.X, position.Y + offset.Y);  # 重载加法运算符，实现位置和偏移量的相加

    public static implicit operator Position(int value) => new(value, value);  # 隐式转换，将整数转换为位置对象

    public override string ToString() => $"{X}{Y}";  # 重写 ToString 方法，返回位置的 X 和 Y 值组成的字符串
}
```