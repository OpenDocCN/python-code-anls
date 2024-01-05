# `d:/src/tocomm/basic-computer-games\56_Life_for_Two\csharp\Piece.cs`

```
using System.Collections.Immutable;  // 导入不可变集合的命名空间
using System.Diagnostics.CodeAnalysis;  // 导入排除分析的命名空间

namespace LifeforTwo;  // 命名空间声明

public struct Piece  // 定义名为Piece的结构体
{
    public const int None = 0x0000;  // 声明名为None的常量，赋值为0x0000
    public const int Player1 = 0x0100;  // 声明名为Player1的常量，赋值为0x0100
    public const int Player2 = 0x1000;  // 声明名为Player2的常量，赋值为0x1000
    private const int PieceMask = Player1 | Player2;  // 声明名为PieceMask的私有常量，赋值为Player1和Player2的按位或结果
    private const int NeighbourValueOffset = 8;  // 声明名为NeighbourValueOffset的私有常量，赋值为8

    private static readonly ImmutableHashSet<int> _willBePlayer1 =  // 声明名为_willBePlayer1的私有静态只读不可变集合，赋值为指定的整数数组转换为不可变集合
        new[] { 0x0003, 0x0102, 0x0103, 0x0120, 0x0130, 0x0121, 0x0112, 0x0111, 0x0012 }.ToImmutableHashSet();
    private static readonly ImmutableHashSet<int> _willBePlayer2 =  // 声明名为_willBePlayer2的私有静态只读不可变集合，赋值为指定的整数数组转换为不可变集合
        new[] { 0x0021, 0x0030, 0x1020, 0x1030, 0x1011, 0x1021, 0x1003, 0x1002, 0x1012 }.ToImmutableHashSet();

    private int _value;  // 声明名为_value的私有整数变量
    private Piece(int value) => _value = value;  # 使用给定的值初始化 Piece 对象的私有属性 _value

    public int Value => _value & PieceMask;  # 获取 Piece 对象的值，并使用位掩码 PieceMask 进行按位与操作

    public bool IsEmpty => (_value & PieceMask) == None;  # 检查 Piece 对象的值是否等于 None，判断是否为空

    public static Piece NewNone() => new(None);  # 创建一个新的 Piece 对象，值为 None
    public static Piece NewPlayer1() => new(Player1);  # 创建一个新的 Piece 对象，值为 Player1
    public static Piece NewPlayer2() => new(Player2);  # 创建一个新的 Piece 对象，值为 Player2

    public Piece AddNeighbour(Piece neighbour)  # 将邻居 Piece 对象的值添加到当前 Piece 对象的值中
    {
        _value += neighbour.Value >> NeighbourValueOffset;  # 将邻居 Piece 对象的值右移 NeighbourValueOffset 位后加到当前 Piece 对象的值中
        return this;  # 返回当前 Piece 对象
    }

    public Piece GetNext() => new(  # 获取下一个状态的 Piece 对象
        _value switch  # 根据当前 Piece 对象的值进行判断
        {
            _ when _willBePlayer1.Contains(_value) => Player1,  # 当当前值在 _willBePlayer1 中时，返回 Player1
            _ when _willBePlayer2.Contains(_value) => Player2,  # 当当前值在 _willBePlayer2 中时，返回 Player2
            _ => None
```
这是一个匿名函数的语法，表示当输入的值不匹配任何已知的模式时，返回 None。

```
        });

    public override string ToString() =>
        (_value & PieceMask) switch
        {
            Player1 => "*",
            Player2 => "#",
            _ => " "
        };
```
这是一个重写 ToString() 方法的语法，根据 Piece 对象的值和特定的模式匹配，返回相应的字符串表示形式。

```
    public static implicit operator Piece(int value) => new(value);
    public static implicit operator int(Piece piece) => piece.Value;
```
这是定义了两个隐式类型转换操作符的语法，允许将 int 类型转换为 Piece 类型，以及将 Piece 类型转换为 int 类型。

```