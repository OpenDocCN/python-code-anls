# `basic-computer-games\75_Roulette\csharp\Bet.cs`

```
# 在 Roulette 命名空间下定义一个内部的记录结构 Bet，包含下注类型、数字和赌注
internal record struct Bet(BetType Type, int Number, int Wager)
{
    # 定义一个公共的属性 Payout，用于计算赔付金额，赔付金额等于赌注乘以下注类型的赔率
    public int Payout => Wager * Type.Payout;
}
```