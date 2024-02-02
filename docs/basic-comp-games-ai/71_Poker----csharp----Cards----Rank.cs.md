# `basic-computer-games\71_Poker\csharp\Cards\Rank.cs`

```py
# 定义一个内部的结构体 Rank，实现了 IComparable 接口
internal struct Rank : IComparable<Rank>
{
    # 定义一个公共的静态属性 Ranks，返回一个包含所有牌面的 Rank 对象的可枚举集合
    public static IEnumerable<Rank> Ranks => new[]
    {
        Two, Three, Four, Five, Six, Seven, Eight, Nine, Ten, Jack, Queen, King, Ace
    };

    # 定义所有牌面的静态属性，每个属性对应一个 Rank 对象
    public static Rank Two = new(2);
    public static Rank Three = new(3);
    public static Rank Four = new(4);
    public static Rank Five = new(5);
    public static Rank Six = new(6);
    public static Rank Seven = new(7);
    public static Rank Eight = new(8);
    public static Rank Nine = new(9);
    public static Rank Ten = new(10);
    public static Rank Jack = new(11, "Jack");
    public static Rank Queen = new(12, "Queen");
    public static Rank King = new(13, "King");
    public static Rank Ace = new(14, "Ace");

    # 定义私有的只读字段 _value 和 _name，分别表示牌面的值和名称
    private readonly int _value;
    private readonly string _name;

    # 定义私有的构造函数，用于创建 Rank 对象，传入牌面的值和名称（可选）
    private Rank(int value, string? name = null)
    {
        _value = value;
        _name = name ?? $" {value} ";
    }

    # 重写 ToString 方法，返回牌面的名称
    public override string ToString() => _name;

    # 实现 IComparable 接口的 CompareTo 方法，用于比较两个 Rank 对象的大小
    public int CompareTo(Rank other) => this - other;

    # 定义 Rank 对象之间的比较运算符重载
    public static bool operator <(Rank x, Rank y) => x._value < y._value;
    public static bool operator >(Rank x, Rank y) => x._value > y._value;
    public static bool operator ==(Rank x, Rank y) => x._value == y._value;
    public static bool operator !=(Rank x, Rank y) => x._value != y._value;

    # 定义 Rank 对象之间的减法运算符重载
    public static int operator -(Rank x, Rank y) => x._value - y._value;

    # 定义 Rank 对象和整数之间的比较运算符重载
    public static bool operator <=(Rank rank, int value) => rank._value <= value;
    public static bool operator >=(Rank rank, int value) => rank._value >= value;

    # 重写 Equals 方法，用于比较两个 Rank 对象是否相等
    public override bool Equals(object? obj) => obj is Rank other && this == other;

    # 重写 GetHashCode 方法，返回 Rank 对象的哈希码
    public override int GetHashCode() => _value.GetHashCode();
}
```