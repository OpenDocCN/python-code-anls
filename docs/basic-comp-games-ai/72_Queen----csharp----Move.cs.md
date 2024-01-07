# `basic-computer-games\72_Queen\csharp\Move.cs`

```

// 命名空间 Queen
namespace Queen;

// 定义一个内部的记录结构 Move，包含对角线和行
internal record struct Move(int Diagonal, int Row)
{
    // 定义静态只读的 Move 对象 Left，表示向左移动
    public static readonly Move Left = new(1, 0);
    // 定义静态只读的 Move 对象 DownLeft，表示向左下移动
    public static readonly Move DownLeft = new(2, 1);
    // 定义静态只读的 Move 对象 Down，表示向下移动
    public static readonly Move Down = new(1, 1);

    // 判断当前 Move 对象是否有效，对角线大于0且为左移、下移或左下移时有效
    public bool IsValid => Diagonal > 0 && (IsLeft || IsDown || IsDownLeft);
    // 判断当前 Move 对象是否为左移
    private bool IsLeft => Row == 0;
    // 判断当前 Move 对象是否为下移
    private bool IsDown => Row == Diagonal;
    // 判断当前 Move 对象是否为左下移
    private bool IsDownLeft => Row * 2 == Diagonal;

    // 重载 * 运算符，实现 Move 对象的缩放
    public static Move operator *(Move move, int scale) => new(move.Diagonal * scale, move.Row * scale);
}

```