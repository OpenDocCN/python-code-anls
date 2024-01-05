# `71_Poker\csharp\Cards\Hand.cs`

```
using System.Text; // 导入 System.Text 命名空间，用于处理字符串和文本数据
using static Poker.Cards.HandRank; // 导入 Poker.Cards.HandRank 命名空间中的所有静态成员
namespace Poker.Cards; // 声明 Poker.Cards 命名空间

internal class Hand // 声明一个内部类 Hand
{
    public static readonly Hand Empty = new Hand(); // 声明一个静态只读的 Hand 对象 Empty，并初始化为一个新的 Hand 对象

    private readonly Card[] _cards; // 声明一个只读的 Card 数组 _cards

    private Hand() // 声明一个私有的无参构造函数
    {
        _cards = Array.Empty<Card>(); // 将 _cards 初始化为空数组
        Rank = None; // 将 Rank 初始化为 None
    }

    public Hand(IEnumerable<Card> cards) // 声明一个公共的构造函数，接受一个 Card 类型的可枚举集合作为参数
        : this(cards, isAfterDraw: false) // 调用另一个构造函数，并传入 cards 和 isAfterDraw 参数
    {
    }
}
    private Hand(IEnumerable<Card> cards, bool isAfterDraw)
    {
        _cards = cards.ToArray(); // 将传入的卡片集合转换为数组并赋值给_cards变量
        (Rank, HighCard, KeepMask) = Analyze(); // 调用Analyze方法分析当前手牌的排名、最高牌和保留掩码，并将结果赋值给对应的属性

        IsWeak = Rank < PartialStraight // 如果排名小于部分顺子，手牌被认为是弱的
            || Rank == PartialStraight && isAfterDraw // 如果排名等于部分顺子并且是抽牌后的手牌，手牌被认为是弱的
            || Rank <= TwoPair && HighCard.Rank <= 6; // 如果排名小于等于两对并且最高牌的点数小于等于6，手牌被认为是弱的
    }

    public string Name => Rank.ToString(HighCard); // 返回手牌的名称，由排名和最高牌组成
    public HandRank Rank { get; } // 获取手牌的排名
    public Card HighCard { get; } // 获取手牌的最高牌
    public int KeepMask { get; set; } // 获取或设置手牌的保留掩码
    public bool IsWeak { get; } // 获取手牌是否弱

    public Hand Replace(int cardNumber, Card newCard)
    {
        if (cardNumber < 1 || cardNumber > _cards.Length) { return this; } // 如果卡片编号不在有效范围内，返回当前手牌

        _cards[cardNumber - 1] = newCard;  # 将新卡片放入指定位置的卡片数组中
        return new Hand(_cards, isAfterDraw: true);  # 返回一个新的手牌对象，表示抽牌后的状态
    }

    private (HandRank, Card, int) Analyze()  # 定义一个私有方法，返回一个元组，包含手牌等级、卡片和整数
    {
        var suitMatchCount = 0;  # 初始化花色匹配计数为0
        for (var i = 0; i < _cards.Length; i++)  # 遍历卡片数组
        {
            if (i < _cards.Length-1 && _cards[i].Suit == _cards[i+1].Suit)  # 如果当前卡片和下一张卡片的花色相同
            {
                suitMatchCount++;  # 花色匹配计数加1
            }
        }
        if (suitMatchCount == 4)  # 如果花色匹配计数为4
        {
            return (Flush, _cards[0], 0b11111);  # 返回一个元组，表示为同花，包含手牌等级、第一张卡片和整数
        }
        var sortedCards = _cards.OrderBy(c => c.Rank).ToArray();  # 对卡片数组按照点数排序，并转换为数组
        // 设置初始手牌等级为Schmaltz
        var handRank = Schmaltz;
        // 初始化保留牌的掩码
        var keepMask = 0;
        // 初始化高牌
        Card highCard = default;
        // 遍历排序后的牌组
        for (var i = 0; i < sortedCards.Length - 1; i++)
        {
            // 检查当前牌与下一张牌是否相同
            var matchesNextCard = sortedCards[i].Rank == sortedCards[i+1].Rank;
            // 检查当前牌与上一张牌是否相同
            var matchesPreviousCard = i > 0 && sortedCards[i].Rank == sortedCards[i - 1].Rank;

            // 如果当前牌与下一张牌相同
            if (matchesNextCard)
            {
                // 更新保留牌的掩码
                keepMask |= 0b11 << i;
                // 更新高牌
                highCard = sortedCards[i];
                // 根据之前的手牌等级和匹配情况更新手牌等级
                handRank = matchesPreviousCard switch
                {
                    // 如果之前的手牌等级小于Pair，则更新为Pair
                    _ when handRank < Pair => Pair,
                    // 如果之前的手牌等级为Pair且匹配上一张牌，则更新为Three
                    true when handRank == Pair => Three,
                    // 如果之前的手牌等级为Pair，则更新为TwoPair
                    _ when handRank == Pair => TwoPair,
                    // 如果之前的手牌等级为TwoPair，则更新为FullHouse
                    _ when handRank == TwoPair => FullHouse,
                    // 如果匹配上一张牌，则更新为Four
                    true => Four,
        // 如果保留的牌的掩码为0，表示没有找到任何牌型，需要继续判断
        if (keepMask == 0)
        {
            // 如果排序后的牌中间三张牌连续，表示有部分顺子
            if (sortedCards[3] - sortedCards[0] == 3)
            {
                // 设置保留的牌的掩码为全1，表示保留所有牌，同时更新手牌等级为部分顺子
                keepMask=0b1111;
                handRank=PartialStraight;
            }
            // 如果排序后的牌后四张牌连续，表示有部分顺子
            if (sortedCards[4] - sortedCards[1] == 3)
            {
                // 如果之前已经有部分顺子，表示找到了顺子，返回顺子牌型和最大牌点数
                if (handRank == PartialStraight)
                {
                    return (Straight, sortedCards[4], 0b11111);
                }
                // 更新手牌等级为部分顺子，设置保留的牌的掩码为0b11110，表示保留除了最小的牌之外的所有牌
                handRank=PartialStraight;
                keepMask=0b11110;
            }
        }
        return handRank < PartialStraight
            ? (Schmaltz, sortedCards[4], 0b11000)
            : (handRank, highCard, keepMask);
    }

    public override string ToString()
    {
        var sb = new StringBuilder();  // 创建一个 StringBuilder 对象，用于构建字符串
        for (var i = 0; i < _cards.Length; i++)  // 遍历 _cards 数组
        {
            var cardDisplay = $" {i+1} --  {_cards[i]}";  // 格式化输出卡牌的序号和值
            // Emulates the effect of the BASIC PRINT statement using the ',' to align text to 14-char print zones
            sb.Append(cardDisplay.PadRight(cardDisplay.Length + 14 - cardDisplay.Length % 14));  // 使用 PadRight 方法在字符串后面填充空格，使得字符串长度达到14的倍数
            if (i % 2 == 1)  // 如果 i 是奇数
            {
                sb.AppendLine();  // 在 StringBuilder 对象中添加换行符
            }
        }
        sb.AppendLine();  // 在 StringBuilder 对象中添加一个额外的换行符
        return sb.ToString();  # 将 StringBuilder 对象转换为字符串并返回

    public static bool operator >(Hand x, Hand y) =>  # 定义大于操作符重载，比较两个 Hand 对象的 Rank 和 HighCard 属性
        x.Rank > y.Rank ||  # 如果 x 的 Rank 大于 y 的 Rank，则返回 true
        x.Rank == y.Rank && x.HighCard > y.HighCard;  # 如果 x 和 y 的 Rank 相等，且 x 的 HighCard 大于 y 的 HighCard，则返回 true

    public static bool operator <(Hand x, Hand y) =>  # 定义小于操作符重载，比较两个 Hand 对象的 Rank 和 HighCard 属性
        x.Rank < y.Rank ||  # 如果 x 的 Rank 小于 y 的 Rank，则返回 true
        x.Rank == y.Rank && x.HighCard < y.HighCard;  # 如果 x 和 y 的 Rank 相等，且 x 的 HighCard 小于 y 的 HighCard，则返回 true
}
```