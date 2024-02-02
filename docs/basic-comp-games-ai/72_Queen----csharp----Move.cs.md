# `basic-computer-games\72_Queen\csharp\Move.cs`

```py
namespace Queen;

// 定义一个命名空间 Queen

internal record struct Move(int Diagonal, int Row)
{
    // 定义一个内部的结构体 Move，包含对角线和行数两个属性

    public static readonly Move Left = new(1, 0);
    // 定义一个静态只读的 Move 对象 Left，对角线为 1，行数为 0
    public static readonly Move DownLeft = new(2, 1);
    // 定义一个静态只读的 Move 对象 DownLeft，对角线为 2，行数为 1
    public static readonly Move Down = new(1, 1);
    // 定义一个静态只读的 Move 对象 Down，对角线为 1，行数为 1

    public bool IsValid => Diagonal > 0 && (IsLeft || IsDown || IsDownLeft);
    // 定义一个公共的只读属性 IsValid，判断对角线大于 0 并且符合左移、下移或左下移的条件

    private bool IsLeft => Row == 0;
    // 定义一个私有的只读属性 IsLeft，判断行数是否为 0
    private bool IsDown => Row == Diagonal;
    // 定义一个私有的只读属性 IsDown，判断行数是否等于对角线
    private bool IsDownLeft => Row * 2 == Diagonal;
    // 定义一个私有的只读属性 IsDownLeft，判断行数的两倍是否等于对角线

    public static Move operator *(Move move, int scale) => new(move.Diagonal * scale, move.Row * scale);
    // 定义一个静态的乘法运算符重载，实现 Move 对象与整数的乘法操作
}
```