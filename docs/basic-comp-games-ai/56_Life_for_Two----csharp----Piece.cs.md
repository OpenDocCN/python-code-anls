# `basic-computer-games\56_Life_for_Two\csharp\Piece.cs`

```
// 使用不可变集合命名空间
using System.Collections.Immutable;
// 使用排除分析命名空间
using System.Diagnostics.CodeAnalysis;

// 声明 LifeforTwo 命名空间
namespace LifeforTwo
{
    // 声明 Piece 结构
    public struct Piece
    {
        // 声明常量 None，表示空
        public const int None = 0x0000;
        // 声明常量 Player1，表示玩家1
        public const int Player1 = 0x0100;
        // 声明常量 Player2，表示玩家2
        public const int Player2 = 0x1000;
        // 声明常量 PieceMask，表示棋子掩码
        private const int PieceMask = Player1 | Player2;
        // 声明常量 NeighbourValueOffset，表示邻居值偏移量
        private const int NeighbourValueOffset = 8;

        // 声明静态只读的将成为 Player1 的不可变哈希集合
        private static readonly ImmutableHashSet<int> _willBePlayer1 = 
            new[] { 0x0003, 0x0102, 0x0103, 0x0120, 0x0130, 0x0121, 0x0112, 0x0111, 0x0012 }.ToImmutableHashSet();
        // 声明静态只读的将成为 Player2 的不可变哈希集合
        private static readonly ImmutableHashSet<int> _willBePlayer2 = 
            new[] { 0x0021, 0x0030, 0x1020, 0x1030, 0x1011, 0x1021, 0x1003, 0x1002, 0x1012 }.ToImmutableHashSet();

        // 声明私有整型变量 _value
        private int _value;

        // 声明私有构造函数，用于初始化 Piece 结构
        private Piece(int value) => _value = value;

        // 声明只读属性 Value，用于获取棋子的值
        public int Value => _value & PieceMask;
        // 声明只读属性 IsEmpty，用于判断棋子是否为空
        public bool IsEmpty => (_value & PieceMask) == None;

        // 声明静态方法 NewNone，用于创建空棋子
        public static Piece NewNone() => new(None);
        // 声明静态方法 NewPlayer1，用于创建玩家1的棋子
        public static Piece NewPlayer1() => new(Player1);
        // 声明静态方法 NewPlayer2，用于创建玩家2的棋子
        public static Piece NewPlayer2() => new(Player2);

        // 声明方法 AddNeighbour，用于添加邻居棋子
        public Piece AddNeighbour(Piece neighbour)
        {
            _value += neighbour.Value >> NeighbourValueOffset;
            return this;
        }

        // 声明方法 GetNext，用于获取下一个状态的棋子
        public Piece GetNext() => new(
            _value switch
            {
                _ when _willBePlayer1.Contains(_value) => Player1,
                _ when _willBePlayer2.Contains(_value) => Player2,
                _ => None
            });

        // 重写 ToString 方法，用于返回棋子的字符串表示
        public override string ToString() =>
            (_value & PieceMask) switch
            {
                Player1 => "*",
                Player2 => "#",
                _ => " "
            };

        // 声明隐式转换，将整型值转换为 Piece 结构
        public static implicit operator Piece(int value) => new(value);
        // 声明隐式转换，将 Piece 结构转换为整型值
        public static implicit operator int(Piece piece) => piece.Value;
    }
}
```