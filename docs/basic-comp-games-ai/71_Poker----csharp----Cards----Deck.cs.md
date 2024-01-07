# `basic-computer-games\71_Poker\csharp\Cards\Deck.cs`

```

// 使用静态引用导入 Rank 类
using static Poker.Cards.Rank;

// 声明 Poker.Cards 命名空间
namespace Poker.Cards;

// 声明 Deck 类，并设置为 internal 访问权限
internal class Deck
{
    // 声明私有只读字段 _cards，用于存储卡牌
    private readonly Card[] _cards;
    // 声明私有字段 _nextCard，用于记录下一张要发的卡牌的索引
    private int _nextCard;

    // 声明 Deck 类的构造函数
    public Deck()
    {
        // 使用 LINQ 生成一副完整的扑克牌
        _cards = Ranks.SelectMany(r => Enum.GetValues<Suit>().Select(s => new Card(r, s))).ToArray();
    }

    // 声明 Shuffle 方法，用于洗牌
    public void Shuffle(IRandom _random)
    {
        // 遍历卡牌数组，随机交换位置
        for (int i = 0; i < _cards.Length; i++)
        {
            var j = _random.Next(_cards.Length);
            (_cards[i], _cards[j]) = (_cards[j], _cards[i]);
        }
        // 重置下一张要发的卡牌的索引
        _nextCard = 0;
    }

    // 声明 DealCard 方法，用于发牌
    public Card DealCard() => _cards[_nextCard++];

    // 声明 DealHand 方法，用于发五张牌组成一手牌
    public Hand DealHand() => new Hand(Enumerable.Range(0, 5).Select(_ => DealCard()));
}

```