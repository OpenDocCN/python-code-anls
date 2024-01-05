# `d:/src/tocomm/basic-computer-games\67_One_Check\csharp\Move.cs`

```
namespace OneCheck;  // 命名空间声明

internal class Move  // 定义一个内部类 Move
{
    public int From { get; init; }  // 定义属性 From，可读可写
    public int To { get; init; }  // 定义属性 To，可读可写
    public int Jumped => (From + To) / 2;  // 定义只读属性 Jumped，返回 From 和 To 的平均值

    public bool IsInRange => From >= 0 && From <= 63 && To >= 0 && To <= 63;  // 定义只读属性 IsInRange，判断 From 和 To 是否在范围内
    public bool IsTwoSpacesDiagonally => RowDelta == 2 && ColumnDelta == 2;  // 定义只读属性 IsTwoSpacesDiagonally，判断是否为对角移动两个空格
    private int RowDelta => Math.Abs(From / 8 - To / 8);  // 定义私有属性 RowDelta，计算行数的差值
    private int ColumnDelta => Math.Abs(From % 8 - To % 8);  // 定义私有属性 ColumnDelta，计算列数的差值
}
```