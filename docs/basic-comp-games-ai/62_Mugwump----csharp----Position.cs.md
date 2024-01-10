# `basic-computer-games\62_Mugwump\csharp\Position.cs`

```
# 在Mugwump命名空间下定义一个内部的记录结构Position，包含X和Y两个浮点数属性
internal record struct Position(float X, float Y)
{
    # 重写ToString方法，返回位置的字符串表示形式
    public override string ToString() => $"( {X} , {Y} )";

    # 定义Position类型的减法运算符，计算两个位置之间的距离并返回Distance类型
    public static Distance operator -(Position p1, Position p2) => new(p1.X - p2.X, p1.Y - p2.Y);
}
```