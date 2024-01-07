# `basic-computer-games\71_Poker\csharp\Cards\Hand.cs`

```

// 使用 System.Text 命名空间
using System.Text;
// 使用 Poker.Cards.HandRank 枚举
using static Poker.Cards.HandRank;
// 声明 Poker.Cards 命名空间
namespace Poker.Cards
{
    // 声明 internal 类 Hand
    internal class Hand
    {
        // 声明静态只读字段 Empty，类型为 Hand
        public static readonly Hand Empty = new Hand();

        // 声明只读字段 _cards，类型为 Card 数组
        private readonly Card[] _cards;

        // 声明私有构造函数 Hand
        private Hand()
        {
            // 初始化 _cards 为空数组
            _cards = Array.Empty<Card>();
            // 初始化 Rank 为 None
            Rank = None;
        }

        // 声明公共构造函数 Hand，接受 IEnumerable<Card> 类型的参数 cards
        public Hand(IEnumerable<Card> cards)
            : this(cards, isAfterDraw: false)
        {
        }

        // 声明私有构造函数 Hand，接受 IEnumerable<Card> 类型的参数 cards 和 bool 类型的参数 isAfterDraw
        private Hand(IEnumerable<Card> cards, bool isAfterDraw)
        {
            // 将 cards 转换为数组并赋值给 _cards
            _cards = cards.ToArray();
            // 调用 Analyze 方法分析手牌，返回结果赋值给 Rank、HighCard 和 KeepMask
            (Rank, HighCard, KeepMask) = Analyze();

            // 根据分析结果判断手牌是否弱牌，赋值给 IsWeak
            IsWeak = Rank < PartialStraight
                || Rank == PartialStraight && isAfterDraw
                || Rank <= TwoPair && HighCard.Rank <= 6;
        }

        // 声明只读属性 Name，返回 Rank.ToString(HighCard)
        public string Name => Rank.ToString(HighCard);
        // 声明属性 Rank，类型为 HandRank
        public HandRank Rank { get; }
        // 声明属性 HighCard，类型为 Card
        public Card HighCard { get; }
        // 声明属性 KeepMask，类型为 int
        public int KeepMask { get; set; }
        // 声明属性 IsWeak，类型为 bool
        public bool IsWeak { get; }

        // 声明方法 Replace，接受 int 类型的参数 cardNumber 和 Card 类型的参数 newCard，返回 Hand 对象
        public Hand Replace(int cardNumber, Card newCard)
        {
            // 如果 cardNumber 不在有效范围内，返回当前 Hand 对象
            if (cardNumber < 1 || cardNumber > _cards.Length) { return this; }

            // 替换指定位置的牌并返回新的 Hand 对象
            _cards[cardNumber - 1] = newCard;
            return new Hand(_cards, isAfterDraw: true);
        }

        // 声明私有方法 Analyze，返回元组类型 (HandRank, Card, int)
        private (HandRank, Card, int) Analyze()
        {
            // ...（具体逻辑略）
        }

        // 重写 ToString 方法
        public override string ToString()
        {
            // ...（具体逻辑略）
        }

        // 声明 > 运算符重载方法，比较两个 Hand 对象的大小
        public static bool operator >(Hand x, Hand y) =>
            x.Rank > y.Rank ||
            x.Rank == y.Rank && x.HighCard > y.HighCard;

        // 声明 < 运算符重载方法，比较两个 Hand 对象的大小
        public static bool operator <(Hand x, Hand y) =>
            x.Rank < y.Rank ||
            x.Rank == y.Rank && x.HighCard < y.HighCard;
    }
}

```