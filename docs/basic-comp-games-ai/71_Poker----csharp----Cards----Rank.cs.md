# `basic-computer-games\71_Poker\csharp\Cards\Rank.cs`

```

// 命名空间声明，定义了Poker.Cards命名空间
namespace Poker.Cards;

// 定义了一个内部结构体Rank，实现了IComparable<Rank>接口
internal struct Rank : IComparable<Rank>
{
    // 定义了一个公共的静态属性Ranks，返回一个包含所有Rank对象的可枚举集合
    public static IEnumerable<Rank> Ranks => new[]
    {
        Two, Three, Four, Five, Six, Seven, Eight, Nine, Ten, Jack, Queen, King, Ace
    };

    // 定义了静态的Rank对象，表示扑克牌的点数，从2到Ace
    public static Rank Two = new(2);
    public static Rank Three = new(3);
    // ... 其他点数的Rank对象

    // 私有的只读字段，表示点数的值和名称
    private readonly int _value;
    private readonly string _name;

    // 私有的构造函数，用于创建Rank对象，传入点数的值和名称（可选）
    private Rank(int value, string? name = null)
    {
        _value = value;
        _name = name ?? $" {value} ";
    }

    // 重写ToString方法，返回点数的名称
    public override string ToString() => _name;

    // 实现IComparable<Rank>接口的CompareTo方法，比较两个Rank对象的大小
    public int CompareTo(Rank other) => this - other;

    // 重载运算符，实现Rank对象之间的比较
    public static bool operator <(Rank x, Rank y) => x._value < y._value;
    // ... 其他比较运算符的重载

    // 重载运算符，实现Rank对象和整数之间的比较
    public static bool operator <=(Rank rank, int value) => rank._value <= value;
    // ... 其他比较运算符的重载

    // 重写Equals方法，判断两个Rank对象是否相等
    public override bool Equals(object? obj) => obj is Rank other && this == other;

    // 重写GetHashCode方法，返回Rank对象的哈希码
    public override int GetHashCode() => _value.GetHashCode();
}

```