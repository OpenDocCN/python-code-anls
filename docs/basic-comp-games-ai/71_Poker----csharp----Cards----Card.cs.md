# `basic-computer-games\71_Poker\csharp\Cards\Card.cs`

```py
# 在Poker.Cards命名空间下定义一个内部的记录结构Card，包含Rank和Suit两个属性
internal record struct Card (Rank Rank, Suit Suit)
{
    # 重写ToString方法，返回卡片的字符串表示形式
    public override string ToString() => $"{Rank} of {Suit}";

    # 定义小于操作符重载，比较两张卡片的Rank属性
    public static bool operator <(Card x, Card y) => x.Rank < y.Rank;
    # 定义大于操作符重载，比较两张卡片的Rank属性
    public static bool operator >(Card x, Card y) => x.Rank > y.Rank;

    # 定义减法操作符重载，返回两张卡片的Rank属性之差
    public static int operator -(Card x, Card y) => x.Rank - y.Rank;
}
```