# `d:/src/tocomm/basic-computer-games\71_Poker\csharp\Cards\HandRank.cs`

```
namespace Poker.Cards;

internal class HandRank
{
    // 定义一个静态的 HandRank 对象，表示没有任何牌型
    public static HandRank None = new(0, "");
    // 定义一个静态的 HandRank 对象，表示 Schmaltz 牌型
    public static HandRank Schmaltz = new(1, "schmaltz, ", c => $"{c.Rank} high");
    // 定义一个静态的 HandRank 对象，表示 PartialStraight 牌型
    public static HandRank PartialStraight = new(2, ""); // 原始代码在这里没有分配显示字符串
    // 定义一个静态的 HandRank 对象，表示一对牌
    public static HandRank Pair = new(3, "a pair of ", c => $"{c.Rank}'s");
    // 定义一个静态的 HandRank 对象，表示两对牌
    public static HandRank TwoPair = new(4, "two pair, ", c => $"{c.Rank}'s");
    // 定义一个静态的 HandRank 对象，表示三张相同的牌
    public static HandRank Three = new(5, "three ", c => $"{c.Rank}'s");
    // 定义一个静态的 HandRank 对象，表示顺子
    public static HandRank Straight = new(6, "straight", c => $"{c.Rank} high");
    // 定义一个静态的 HandRank 对象，表示同花
    public static HandRank Flush = new(7, "a flush in ", c => c.Suit.ToString());
    // 定义一个静态的 HandRank 对象，表示葫芦
    public static HandRank FullHouse = new(8, "full house, ", c => $"{c.Rank}'s");
    // 定义一个静态的 HandRank 对象，表示四张相同的牌
    public static HandRank Four = new(9, "four ", c => $"{c.Rank}'s");
    // 原始代码没有检测同花顺或皇家同花顺

    private readonly int _value; // 存储牌型的值
    private readonly string _displayName; // 存储牌型的显示名称
    private readonly Func<Card, string> _suffixSelector; // 存储后缀选择器
    private HandRank(int value, string displayName, Func<Card, string>? suffixSelector = null)
    {
        // 初始化 HandRank 对象的值、显示名称和后缀选择器
        _value = value;
        _displayName = displayName;
        _suffixSelector = suffixSelector ?? (_ => "");
    }

    // 重写 ToString 方法，返回 HandRank 对象的显示名称和后缀选择器选择的内容
    public string ToString(Card highCard) => $"{_displayName}{_suffixSelector.Invoke(highCard)}";

    // 定义 HandRank 对象之间的大于比较操作
    public static bool operator >(HandRank x, HandRank y) => x._value > y._value;
    // 定义 HandRank 对象之间的小于比较操作
    public static bool operator <(HandRank x, HandRank y) => x._value < y._value;
    // 定义 HandRank 对象之间的大于等于比较操作
    public static bool operator >=(HandRank x, HandRank y) => x._value >= y._value;
    // 定义 HandRank 对象之间的小于等于比较操作
    public static bool operator <=(HandRank x, HandRank y) => x._value <= y._value;
}
```