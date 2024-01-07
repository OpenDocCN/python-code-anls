# `basic-computer-games\71_Poker\csharp\Cards\HandRank.cs`

```

// 命名空间为Poker.Cards，表示该类属于Poker.Cards命名空间
internal class HandRank
{
    // 定义静态字段None，表示没有特殊牌型，值为0，显示字符串为空
    public static HandRank None = new(0, "");
    // 定义静态字段Schmaltz，表示一对牌，值为1，显示字符串为"schmaltz, "，并根据最高牌的等级生成显示字符串
    public static HandRank Schmaltz = new(1, "schmaltz, ", c => $"{c.Rank} high");
    // 定义静态字段PartialStraight，表示部分顺子，值为2，显示字符串为空。原始代码在此处没有分配显示字符串
    public static HandRank PartialStraight = new(2, ""); 
    // 定义静态字段Pair，表示一对牌，值为3，显示字符串为"a pair of "，并根据最高牌的等级生成显示字符串
    public static HandRank Pair = new(3, "a pair of ", c => $"{c.Rank}'s");
    // 定义静态字段TwoPair，表示两对牌，值为4，显示字符串为"two pair, "，并根据最高牌的等级生成显示字符串
    public static HandRank TwoPair = new(4, "two pair, ", c => $"{c.Rank}'s");
    // 定义静态字段Three，表示三张相同的牌，值为5，显示字符串为"three "，并根据最高牌的等级生成显示字符串
    public static HandRank Three = new(5, "three ", c => $"{c.Rank}'s");
    // 定义静态字段Straight，表示顺子，值为6，显示字符串为"straight"，并根据最高牌的等级生成显示字符串
    public static HandRank Straight = new(6, "straight", c => $"{c.Rank} high");
    // 定义静态字段Flush，表示同花，值为7，显示字符串为"a flush in "，并根据花色生成显示字符串
    public static HandRank Flush = new(7, "a flush in ", c => c.Suit.ToString());
    // 定义静态字段FullHouse，表示葫芦，值为8，显示字符串为"full house, "，并根据最高牌的等级生成显示字符串
    public static HandRank FullHouse = new(8, "full house, ", c => $"{c.Rank}'s");
    // 定义静态字段Four，表示四张相同的牌，值为9，显示字符串为"four "，并根据最高牌的等级生成显示字符串
    public static HandRank Four = new(9, "four ", c => $"{c.Rank}'s");
    // 原始代码没有检测同花顺或皇家同花顺

    // 私有字段，存储牌型的值
    private readonly int _value;
    // 私有字段，存储牌型的显示字符串
    private readonly string _displayName;
    // 私有字段，存储用于生成显示字符串的方法
    private readonly Func<Card, string> _suffixSelector;

    // 构造函数，初始化牌型的值、显示字符串和用于生成显示字符串的方法
    private HandRank(int value, string displayName, Func<Card, string>? suffixSelector = null)
    {
        _value = value;
        _displayName = displayName;
        _suffixSelector = suffixSelector ?? (_ => "");
    }

    // 重写ToString方法，根据最高牌生成完整的显示字符串
    public string ToString(Card highCard) => $"{_displayName}{_suffixSelector.Invoke(highCard)}";

    // 重载比较运算符，用于比较两个牌型的大小
    public static bool operator >(HandRank x, HandRank y) => x._value > y._value;
    public static bool operator <(HandRank x, HandRank y) => x._value < y._value;
    public static bool operator >=(HandRank x, HandRank y) => x._value >= y._value;
    public static bool operator <=(HandRank x, HandRank y) => x._value <= y._value;
}

```