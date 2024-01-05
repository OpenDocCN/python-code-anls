# `71_Poker\csharp\Cards\Deck.cs`

```
using static Poker.Cards.Rank;  # 导入Poker.Cards.Rank命名空间中的所有静态成员

namespace Poker.Cards;  # 声明Poker.Cards命名空间

internal class Deck  # 声明一个内部类Deck
{
    private readonly Card[] _cards;  # 声明一个只读的Card类型数组_cards
    private int _nextCard;  # 声明一个私有的整型变量_nextCard

    public Deck()  # 声明一个公共的构造函数Deck
    {
        _cards = Ranks.SelectMany(r => Enum.GetValues<Suit>().Select(s => new Card(r, s))).ToArray();  # 使用LINQ语句生成一副扑克牌的所有组合，并存储在_cards数组中
    }

    public void Shuffle(IRandom _random)  # 声明一个公共的无返回值的Shuffle方法，接受一个IRandom类型的参数_random
    {
        for (int i = 0; i < _cards.Length; i++)  # 循环_cards数组的长度次
        {
            var j = _random.Next(_cards.Length);  # 生成一个随机数j，范围在0到_cards数组的长度之间
            (_cards[i], _cards[j]) = (_cards[j], _cards[i]);  # 交换_cards数组中索引为i和j的元素
        }
        _nextCard = 0;  # 重置下一张要发的卡牌的索引为0
    }

    public Card DealCard() => _cards[_nextCard++];  # 发一张卡牌并返回，同时将下一张要发的卡牌索引加1

    public Hand DealHand() => new Hand(Enumerable.Range(0, 5).Select(_ => DealCard()));  # 发一手牌（5张），通过调用DealCard()方法来获取每张牌
}
```