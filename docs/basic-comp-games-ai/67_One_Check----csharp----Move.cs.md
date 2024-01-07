# `basic-computer-games\67_One_Check\csharp\Move.cs`

```

namespace OneCheck;

// 定义一个内部类 Move
internal class Move
{
    // 定义属性 From，表示起始位置
    public int From { get; init; }
    // 定义属性 To，表示目标位置
    public int To { get; init; }
    // 定义只读属性 Jumped，表示跳跃位置
    public int Jumped => (From + To) / 2;

    // 定义只读属性 IsInRange，表示移动是否在范围内
    public bool IsInRange => From >= 0 && From <= 63 && To >= 0 && To <= 63;
    // 定义只读属性 IsTwoSpacesDiagonally，表示移动是否为对角线上的两步
    public bool IsTwoSpacesDiagonally => RowDelta == 2 && ColumnDelta == 2;
    // 定义私有属性 RowDelta，表示行的变化量
    private int RowDelta => Math.Abs(From / 8 - To / 8);
    // 定义私有属性 ColumnDelta，表示列的变化量
    private int ColumnDelta => Math.Abs(From % 8 - To % 8);
}

```