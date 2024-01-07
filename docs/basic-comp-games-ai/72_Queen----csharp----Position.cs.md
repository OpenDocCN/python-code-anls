# `basic-computer-games\72_Queen\csharp\Position.cs`

```

namespace Queen;

// 定义一个命名空间 Queen

internal record struct Position(int Diagonal, int Row)
{
    // 定义一个内部的记录结构 Position，包含对角线和行两个整数属性

    // 定义一个静态的 Position 对象 Zero，对角线和行都为 0
    public static readonly Position Zero = new(0);

    // 构造函数，根据一个整数创建 Position 对象
    public Position(int number)
        : this(Diagonal: number / 10, Row: number % 10)
    {
    }

    // 判断当前 Position 对象是否为零
    public bool IsZero => Row == 0 && Diagonal == 0;
    // 判断当前 Position 对象是否为起点
    public bool IsStart => Row == 1 || Row == Diagonal;
    // 判断当前 Position 对象是否为终点
    public bool IsEnd => Row == 8 && Diagonal == 15;

    // 重写 ToString 方法，返回 Position 对象的字符串表示
    public override string ToString() => $"{Diagonal}{Row}";

    // 定义一个隐式转换，将整数转换为 Position 对象
    public static implicit operator Position(int value) => new(value);

    // 定义 Position 对象和 Move 对象的加法运算
    public static Position operator +(Position position, Move move)
        => new(Diagonal: position.Diagonal + move.Diagonal, Row: position.Row + move.Row);
    // 定义 Position 对象和 Position 对象的减法运算，返回一个 Move 对象
    public static Move operator -(Position to, Position from)
        => new(Diagonal: to.Diagonal - from.Diagonal, Row: to.Row - from.Row);
}

```