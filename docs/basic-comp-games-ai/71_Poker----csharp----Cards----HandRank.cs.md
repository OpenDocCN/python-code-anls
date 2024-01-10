# `basic-computer-games\71_Poker\csharp\Cards\HandRank.cs`

```
// 定义一个命名空间为 Poker.Cards 的内部类 HandRank
internal class HandRank
{
    // 定义一个公共静态 HandRank 对象 None，赋值为一个新的 HandRank 对象，参数为 0 和空字符串
    public static HandRank None = new(0, "");
    // 定义一个公共静态 HandRank 对象 Schmaltz，赋值为一个新的 HandRank 对象，参数为 1、"schmaltz, " 和一个函数，用于返回牌面值最大的牌
    public static HandRank Schmaltz = new(1, "schmaltz, ", c => $"{c.Rank} high");
    // 定义一个公共静态 HandRank 对象 PartialStraight，赋值为一个新的 HandRank 对象，参数为 2 和空字符串
    public static HandRank PartialStraight = new(2, ""); // The original code does not assign a display string here
    // 定义一个公共静态 HandRank 对象 Pair，赋值为一个新的 HandRank 对象，参数为 3、"a pair of " 和一个函数，用于返回一对牌的牌面值
    public static HandRank Pair = new(3, "a pair of ", c => $"{c.Rank}'s");
    // 定义一个公共静态 HandRank 对象 TwoPair，赋值为一个新的 HandRank 对象，参数为 4、"two pair, " 和一个函数，用于返回两对牌的牌面值
    public static HandRank TwoPair = new(4, "two pair, ", c => $"{c.Rank}'s");
    // 定义一个公共静态 HandRank 对象 Three，赋值为一个新的 HandRank 对象，参数为 5、"three " 和一个函数，用于返回三张相同牌面值的牌
    public static HandRank Three = new(5, "three ", c => $"{c.Rank}'s");
    // 定义一个公共静态 HandRank 对象 Straight，赋值为一个新的 HandRank 对象，参数为 6、"straight" 和一个函数，用于返回顺子的最大牌面值
    public static HandRank Straight = new(6, "straight", c => $"{c.Rank} high");
    // 定义一个公共静态 HandRank 对象 Flush，赋值为一个新的 HandRank 对象，参数为 7、"a flush in " 和一个函数，用于返回同花的花色
    public static HandRank Flush = new(7, "a flush in ", c => c.Suit.ToString());
    // 定义一个公共静态 HandRank 对象 FullHouse，赋值为一个新的 HandRank 对象，参数为 8、"full house, " 和一个函数，用于返回葫芦中三张相同牌面值的牌
    public static HandRank FullHouse = new(8, "full house, ", c => $"{c.Rank}'s");
    // 定义一个公共静态 HandRank 对象 Four，赋值为一个新的 HandRank 对象，参数为 9 和一个函数，用于返回四张相同牌面值的牌
    public static HandRank Four = new(9, "four ", c => $"{c.Rank}'s");
    // 原始代码没有检测顺子同花顺或皇家同花顺

    // 定义一个只读整型字段 _value
    private readonly int _value;
    // 定义一个只读字符串字段 _displayName
    private readonly string _displayName;
    // 定义一个只读 Func 字段 _suffixSelector，用于选择后缀
    private readonly Func<Card, string> _suffixSelector;

    // 定义一个私有构造函数，参数为整型 value、字符串 displayName 和可空的函数 suffixSelector
    private HandRank(int value, string displayName, Func<Card, string>? suffixSelector = null)
    {
        // 将参数 value 赋值给 _value
        _value = value;
        // 将参数 displayName 赋值给 _displayName
        _displayName = displayName;
        // 如果 suffixSelector 不为空，则赋值给 _suffixSelector，否则赋值一个函数，用于返回空字符串
        _suffixSelector = suffixSelector ?? (_ => "");
    }

    // 定义一个方法 ToString，参数为高牌 highCard，用于返回组合后的显示字符串
    public string ToString(Card highCard) => $"{_displayName}{_suffixSelector.Invoke(highCard)}";

    // 定义一个重载运算符 >，用于比较两个 HandRank 对象的大小
    public static bool operator >(HandRank x, HandRank y) => x._value > y._value;
    // 定义一个重载运算符 <，用于比较两个 HandRank 对象的大小
    public static bool operator <(HandRank x, HandRank y) => x._value < y._value;
    // 定义一个重载运算符 >=，用于比较两个 HandRank 对象的大小
    public static bool operator >=(HandRank x, HandRank y) => x._value >= y._value;
    // 定义一个重载运算符 <=，用于比较两个 HandRank 对象的大小
    public static bool operator <=(HandRank x, HandRank y) => x._value <= y._value;
}
```