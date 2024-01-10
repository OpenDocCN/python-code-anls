# `basic-computer-games\72_Queen\csharp\Position.cs`

```
# 命名空间 Queen 下的内部记录结构 Position，包含对角线和行两个整数属性
internal record struct Position(int Diagonal, int Row)
{
    # 静态只读属性 Zero，表示位置 (0, 0)
    public static readonly Position Zero = new(0);

    # 构造函数，根据一个整数创建位置对象
    public Position(int number)
        : this(Diagonal: number / 10, Row: number % 10)
    {
    }

    # 属性 IsZero，表示位置是否为 (0, 0)
    public bool IsZero => Row == 0 && Diagonal == 0;
    # 属性 IsStart，表示位置是否为起始位置
    public bool IsStart => Row == 1 || Row == Diagonal;
    # 属性 IsEnd，表示位置是否为结束位置
    public bool IsEnd => Row == 8 && Diagonal == 15;

    # 重写 ToString 方法，返回位置的字符串表示
    public override string ToString() => $"{Diagonal}{Row}";

    # 隐式转换，将整数转换为位置对象
    public static implicit operator Position(int value) => new(value);

    # 重载 + 运算符，实现位置对象和移动对象的相加
    public static Position operator +(Position position, Move move)
        => new(Diagonal: position.Diagonal + move.Diagonal, Row: position.Row + move.Row);
    # 重载 - 运算符，实现位置对象和位置对象的相减，返回移动对象
    public static Move operator -(Position to, Position from)
        => new(Diagonal: to.Diagonal - from.Diagonal, Row: to.Row - from.Row);
}
```