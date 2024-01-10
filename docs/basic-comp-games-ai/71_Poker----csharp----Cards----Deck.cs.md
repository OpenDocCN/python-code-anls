# `basic-computer-games\71_Poker\csharp\Cards\Deck.cs`

```
using static Poker.Cards.Rank;
// 导入 Rank 枚举的静态成员

namespace Poker.Cards;
// 声明命名空间 Poker.Cards

internal class Deck
// 声明内部类 Deck
{
    private readonly Card[] _cards;
    // 声明只读字段 _cards，类型为 Card 数组
    private int _nextCard;
    // 声明字段 _nextCard，类型为整数

    public Deck()
    // 声明构造函数 Deck
    {
        _cards = Ranks.SelectMany(r => Enum.GetValues<Suit>().Select(s => new Card(r, s))).ToArray();
        // 使用 Ranks 枚举的每个元素和 Suit 枚举的每个元素创建 Card 对象数组，并赋值给 _cards
    }

    public void Shuffle(IRandom _random)
    // 声明方法 Shuffle，参数为 IRandom 类型的 _random
    {
        for (int i = 0; i < _cards.Length; i++)
        // 循环，i 从 0 到 _cards.Length - 1
        {
            var j = _random.Next(_cards.Length);
            // 生成一个随机数 j，范围为 0 到 _cards.Length - 1
            (_cards[i], _cards[j]) = (_cards[j], _cards[i]);
            // 交换 _cards[i] 和 _cards[j] 的值
        }
        _nextCard = 0;
        // 将 _nextCard 置为 0
    }

    public Card DealCard() => _cards[_nextCard++];
    // 声明方法 DealCard，返回 _cards[_nextCard]，并将 _nextCard 加一

    public Hand DealHand() => new Hand(Enumerable.Range(0, 5).Select(_ => DealCard()));
    // 声明方法 DealHand，返回一个新的 Hand 对象，其中包含 5 张牌，使用 DealCard 方法来获取
}
```