# `basic-computer-games\67_One_Check\csharp\Move.cs`

```
internal class Move
{
    // 表示棋子移动的起始位置
    public int From { get; init; }
    // 表示棋子移动的目标位置
    public int To { get; init; }
    // 计算跳跃位置，即起始位置和目标位置的中间位置
    public int Jumped => (From + To) / 2;

    // 判断移动是否在合理范围内
    public bool IsInRange => From >= 0 && From <= 63 && To >= 0 && To <= 63;
    // 判断移动是否为对角线上的两步
    public bool IsTwoSpacesDiagonally => RowDelta == 2 && ColumnDelta == 2;
    // 计算行数的变化
    private int RowDelta => Math.Abs(From / 8 - To / 8);
    // 计算列数的变化
    private int ColumnDelta => Math.Abs(From % 8 - To % 8);
}
```