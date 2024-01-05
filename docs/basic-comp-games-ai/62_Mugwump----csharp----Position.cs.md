# `62_Mugwump\csharp\Position.cs`

```
namespace Mugwump;  // 命名空间声明，用于组织和管理代码

internal record struct Position(float X, float Y)  // 定义一个名为 Position 的结构体，包含两个 float 类型的字段 X 和 Y
{
    public override string ToString() => $"( {X} , {Y} )";  // 重写 ToString 方法，返回包含 X 和 Y 值的字符串

    public static Distance operator -(Position p1, Position p2) => new(p1.X - p2.X, p1.Y - p2.Y);  // 定义减法运算符重载，返回两个 Position 对象之间的距离
}
```