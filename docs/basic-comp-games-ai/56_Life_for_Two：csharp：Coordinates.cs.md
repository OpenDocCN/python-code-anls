# `d:/src/tocomm/basic-computer-games\56_Life_for_Two\csharp\Coordinates.cs`

```
namespace LifeforTwo;  # 命名空间声明

internal record Coordinates (int X, int Y)  # 定义一个内部记录类型 Coordinates，包含两个整型字段 X 和 Y

{
    public static bool TryCreate((float X, float Y) values, out Coordinates coordinates)  # 定义一个静态方法，尝试创建 Coordinates 对象，并返回是否成功的布尔值
    {
        if (values.X <= 0 || values.X > 5 || values.Y <= 0 || values.Y > 5)  # 如果传入的值不在指定范围内
        {
            coordinates = new(0, 0);  # 创建一个默认的 Coordinates 对象
            return false;  # 返回失败
        }

        coordinates = new((int)values.X, (int)values.Y);  # 根据传入的值创建 Coordinates 对象
        return true;  # 返回成功
    }

    public static Coordinates operator +(Coordinates coordinates, int value) =>  # 定义一个重载的加法运算符，使得 Coordinates 对象可以与整数相加
        new (coordinates.X + value, coordinates.Y + value);  # 返回相加后的 Coordinates 对象

    public IEnumerable<Coordinates> GetNeighbors()  # 定义一个方法，返回当前坐标的邻居坐标集合
    # 返回当前位置上、下、左、右、左上、右上、左下、右下八个相邻位置的坐标
    {
        # 返回当前位置的左边位置
        yield return new(X - 1, Y);
        # 返回当前位置的右边位置
        yield return new(X + 1, Y);
        # 返回当前位置的上边位置
        yield return new(X, Y - 1);
        # 返回当前位置的下边位置
        yield return new(X, Y + 1);
        # 返回当前位置的左上位置
        yield return new(X - 1, Y - 1);
        # 返回当前位置的右上位置
        yield return new(X + 1, Y - 1);
        # 返回当前位置的左下位置
        yield return new(X - 1, Y + 1);
        # 返回当前位置的右下位置
        yield return new(X + 1, Y + 1);
    }
}
```