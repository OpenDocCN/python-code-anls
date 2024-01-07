# `basic-computer-games\56_Life_for_Two\csharp\Piece.cs`

```

// 使用不可变集合的命名空间
using System.Collections.Immutable;
// 使用可空引用类型的命名空间
using System.Diagnostics.CodeAnalysis;

// 定义 LifeforTwo 命名空间
namespace LifeforTwo
{
    // 定义 Piece 结构
    public struct Piece
    {
        // 定义常量 None，Player1，Player2
        public const int None = 0x0000;
        public const int Player1 = 0x0100;
        public const int Player2 = 0x1000;
        // 定义 PieceMask 和 NeighbourValueOffset 常量
        private const int PieceMask = Player1 | Player2;
        private const int NeighbourValueOffset = 8;

        // 定义将要成为 Player1 和 Player2 的不可变集合
        private static readonly ImmutableHashSet<int> _willBePlayer1 = 
            new[] { 0x0003, 0x0102, 0x0103, 0x0120, 0x0130, 0x0121, 0x0112, 0x0111, 0x0012 }.ToImmutableHashSet();
        private static readonly ImmutableHashSet<int> _willBePlayer2 = 
            new[] { 0x0021, 0x0030, 0x1020, 0x1030, 0x1011, 0x1021, 0x1003, 0x1002, 0x1012 }.ToImmutableHashSet();

        // 定义私有变量 _value
        private int _value;

        // 定义私有构造函数
        private Piece(int value) => _value = value;

        // 定义 Value 和 IsEmpty 属性
        public int Value => _value & PieceMask;
        public bool IsEmpty => (_value & PieceMask) == None;

        // 定义 NewNone，NewPlayer1，NewPlayer2 方法
        public static Piece NewNone() => new(None);
        public static Piece NewPlayer1() => new(Player1);
        public static Piece NewPlayer2() => new(Player2);

        // 定义 AddNeighbour 方法
        public Piece AddNeighbour(Piece neighbour)
        {
            _value += neighbour.Value >> NeighbourValueOffset;
            return this;
        }

        // 定义 GetNext 方法
        public Piece GetNext() => new(
            _value switch
            {
                _ when _willBePlayer1.Contains(_value) => Player1,
                _ when _willBePlayer2.Contains(_value) => Player2,
                _ => None
            });

        // 重写 ToString 方法
        public override string ToString() =>
            (_value & PieceMask) switch
            {
                Player1 => "*",
                Player2 => "#",
                _ => " "
            };

        // 定义隐式转换操作符
        public static implicit operator Piece(int value) => new(value);
        public static implicit operator int(Piece piece) => piece.Value;
    }
}

```