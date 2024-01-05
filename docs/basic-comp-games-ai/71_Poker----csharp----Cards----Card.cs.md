# `71_Poker\csharp\Cards\Card.cs`

```
namespace Poker.Cards;  // 声明命名空间为Poker.Cards，用于组织和管理代码

internal record struct Card (Rank Rank, Suit Suit)  // 声明一个内部的记录结构体Card，包含Rank和Suit两个属性
{
    public override string ToString() => $"{Rank} of {Suit}";  // 重写ToString方法，返回卡牌的字符串表示形式

    public static bool operator <(Card x, Card y) => x.Rank < y.Rank;  // 定义小于操作符，比较卡牌的Rank属性
    public static bool operator >(Card x, Card y) => x.Rank > y.Rank;  // 定义大于操作符，比较卡牌的Rank属性

    public static int operator -(Card x, Card y) => x.Rank - y.Rank;  // 定义减法操作符，返回卡牌的Rank属性之差
}
```