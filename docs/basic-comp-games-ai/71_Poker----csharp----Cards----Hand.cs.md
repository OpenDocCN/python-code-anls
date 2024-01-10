# `basic-computer-games\71_Poker\csharp\Cards\Hand.cs`

```
using System.Text;
using static Poker.Cards.HandRank;
namespace Poker.Cards;

internal class Hand
{
    // 创建一个空的 Hand 对象
    public static readonly Hand Empty = new Hand();

    // 私有字段，存储手牌的数组
    private readonly Card[] _cards;

    // 私有构造函数，用于创建空的 Hand 对象
    private Hand()
    {
        // 将 _cards 初始化为空数组
        _cards = Array.Empty<Card>();
        // 将 Rank 初始化为 None
        Rank = None;
    }

    // 公共构造函数，接受一组卡牌作为参数
    public Hand(IEnumerable<Card> cards)
        : this(cards, isAfterDraw: false)
    {
    }

    // 私有构造函数，接受一组卡牌和一个布尔值作为参数
    private Hand(IEnumerable<Card> cards, bool isAfterDraw)
    {
        // 将传入的卡牌转换为数组并赋值给 _cards
        _cards = cards.ToArray();
        // 调用 Analyze 方法分析手牌，将结果赋值给 Rank, HighCard 和 KeepMask
        (Rank, HighCard, KeepMask) = Analyze();

        // 根据分析结果判断手牌是否弱
        IsWeak = Rank < PartialStraight
            || Rank == PartialStraight && isAfterDraw
            || Rank <= TwoPair && HighCard.Rank <= 6;
    }

    // 公共属性，返回手牌的名称
    public string Name => Rank.ToString(HighCard);
    // 公共属性，返回手牌的等级
    public HandRank Rank { get; }
    // 公共属性，返回手牌中最大的卡牌
    public Card HighCard { get; }
    // 公共属性，获取或设置保留卡牌的掩码
    public int KeepMask { get; set; }
    // 公共属性，返回手牌是否弱
    public bool IsWeak { get; }

    // 公共方法，替换手牌中的一张卡牌
    public Hand Replace(int cardNumber, Card newCard)
    {
        // 如果卡牌编号不在有效范围内，则返回当前手牌
        if (cardNumber < 1 || cardNumber > _cards.Length) { return this; }

        // 替换指定位置的卡牌，并返回一个新的 Hand 对象
        _cards[cardNumber - 1] = newCard;
        return new Hand(_cards, isAfterDraw: true);
    }

    // 私有方法，分析手牌并返回结果
    private (HandRank, Card, int) Analyze()
    {
        // 初始化花色匹配计数
        var suitMatchCount = 0;
        // 遍历卡牌数组
        for (var i = 0; i < _cards.Length; i++)
        {
            // 检查当前卡牌与下一张卡牌的花色是否相同
            if (i < _cards.Length-1 && _cards[i].Suit == _cards[i+1].Suit)
            {
                // 如果相同，花色匹配计数加一
                suitMatchCount++;
            }
        }
        // 如果花色匹配计数为4，表示为同花，返回同花的牌型和相关信息
        if (suitMatchCount == 4)
        {
            return (Flush, _cards[0], 0b11111);
        }
        // 对卡牌数组按照点数排序
        var sortedCards = _cards.OrderBy(c => c.Rank).ToArray();
    
        // 初始化手牌等级、保留掩码和最高点数的卡牌
        var handRank = Schmaltz;
        var keepMask = 0;
        Card highCard = default;
        // 遍历排序后的卡牌数组
        for (var i = 0; i < sortedCards.Length - 1; i++)
        {
            // 检查当前卡牌与下一张卡牌的点数是否相同
            var matchesNextCard = sortedCards[i].Rank == sortedCards[i+1].Rank;
            // 检查当前卡牌与上一张卡牌的点数是否相同
            var matchesPreviousCard = i > 0 && sortedCards[i].Rank == sortedCards[i - 1].Rank;
    
            // 如果当前卡牌与下一张卡牌的点数相同
            if (matchesNextCard)
            {
                // 更新保留掩码和最高点数的卡牌，并根据情况更新手牌等级
                keepMask |= 0b11 << i;
                highCard = sortedCards[i];
                handRank = matchesPreviousCard switch
                {
                    _ when handRank < Pair => Pair,
                    true when handRank == Pair => Three,
                    _ when handRank == Pair => TwoPair,
                    _ when handRank == TwoPair => FullHouse,
                    true => Four,
                    _ => FullHouse
                };
            }
        }
        // 如果没有相同点数的卡牌
        if (keepMask == 0)
        {
            // 如果卡牌点数连续，表示为部分顺子
            if (sortedCards[3] - sortedCards[0] == 3)
            {
                keepMask=0b1111;
                handRank=PartialStraight;
            }
            // 如果前四张卡牌点数连续，表示为部分顺子
            if (sortedCards[4] - sortedCards[1] == 3)
            {
                // 如果之前已经判断为部分顺子，返回顺子的牌型和相关信息
                if (handRank == PartialStraight)
                {
                    return (Straight, sortedCards[4], 0b11111);
                }
                // 更新手牌等级和保留掩码
                handRank=PartialStraight;
                keepMask=0b11110;
            }
        }
        // 根据手牌等级返回相应的牌型和相关信息
        return handRank < PartialStraight
            ? (Schmaltz, sortedCards[4], 0b11000)
            : (handRank, highCard, keepMask);
    }
    
    public override string ToString()
    {
        // 创建一个 StringBuilder 对象，用于构建字符串
        var sb = new StringBuilder();
        // 遍历 _cards 数组
        for (var i = 0; i < _cards.Length; i++)
        {
            // 根据卡片信息创建 cardDisplay 字符串
            var cardDisplay = $" {i+1} --  {_cards[i]}";
            // 使用 PadRight 方法模拟 BASIC PRINT 语句的效果，将文本对齐到14个字符的打印区域
            sb.Append(cardDisplay.PadRight(cardDisplay.Length + 14 - cardDisplay.Length % 14));
            // 如果 i 是偶数，换行
            if (i % 2 == 1)
            {
                sb.AppendLine();
            }
        }
        // 在最后添加一个空行
        sb.AppendLine();
        // 返回 StringBuilder 对象中的字符串
        return sb.ToString();
    }
    
    // 定义大于操作符重载
    public static bool operator >(Hand x, Hand y) =>
        x.Rank > y.Rank ||  // 如果 x 的牌型大于 y 的牌型，返回 true
        x.Rank == y.Rank && x.HighCard > y.HighCard;  // 如果牌型相同，比较最大的牌，返回 true 或 false
    
    // 定义小于操作符重载
    public static bool operator <(Hand x, Hand y) =>
        x.Rank < y.Rank ||  // 如果 x 的牌型小于 y 的牌型，返回 true
        x.Rank == y.Rank && x.HighCard < y.HighCard;  // 如果牌型相同，比较最大的牌，返回 true 或 false
    }
# 闭合前面的函数定义
```