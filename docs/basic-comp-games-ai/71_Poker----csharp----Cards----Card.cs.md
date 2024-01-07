# `basic-computer-games\71_Poker\csharp\Cards\Card.cs`

```

// 命名空间声明，表示该类属于Poker.Cards命名空间
namespace Poker.Cards;

// 内部记录结构体，表示一张扑克牌，包含花色和点数
internal record struct Card (Rank Rank, Suit Suit)
{
    // 重写ToString方法，返回该扑克牌的点数和花色
    public override string ToString() => $"{Rank} of {Suit}";

    // 定义小于操作符重载，比较两张扑克牌的点数大小
    public static bool operator <(Card x, Card y) => x.Rank < y.Rank;
    // 定义大于操作符重载，比较两张扑克牌的点数大小
    public static bool operator >(Card x, Card y) => x.Rank > y.Rank;

    // 定义减法操作符重载，计算两张扑克牌的点数差
    public static int operator -(Card x, Card y) => x.Rank - y.Rank;
}

```