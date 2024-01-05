# `72_Queen\csharp\Move.cs`

```
namespace Queen;  // 命名空间声明

internal record struct Move(int Diagonal, int Row)  // 定义一个名为 Move 的结构体，包含两个整型参数 Diagonal 和 Row
{
    public static readonly Move Left = new(1, 0);  // 创建一个名为 Left 的静态只读 Move 对象，参数为 (1, 0)
    public static readonly Move DownLeft = new(2, 1);  // 创建一个名为 DownLeft 的静态只读 Move 对象，参数为 (2, 1)
    public static readonly Move Down = new(1, 1);  // 创建一个名为 Down 的静态只读 Move 对象，参数为 (1, 1)

    public bool IsValid => Diagonal > 0 && (IsLeft || IsDown || IsDownLeft);  // 定义一个名为 IsValid 的属性，判断是否为有效的移动
    private bool IsLeft => Row == 0;  // 定义一个名为 IsLeft 的私有属性，判断是否为左移
    private bool IsDown => Row == Diagonal;  // 定义一个名为 IsDown 的私有属性，判断是否为下移
    private bool IsDownLeft => Row * 2 == Diagonal;  // 定义一个名为 IsDownLeft 的私有属性，判断是否为左下移

    public static Move operator *(Move move, int scale) => new(move.Diagonal * scale, move.Row * scale);  // 定义一个名为 * 的运算符重载，实现 Move 对象与整数的乘法
}
```