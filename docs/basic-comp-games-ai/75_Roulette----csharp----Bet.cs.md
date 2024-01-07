# `basic-computer-games\75_Roulette\csharp\Bet.cs`

```

// 命名空间为Roulette
namespace Roulette;

// 内部的记录结构体Bet，包含下注类型、数字和赌注
internal record struct Bet(BetType Type, int Number, int Wager)
{
    // 获取赔付金额，赔付金额为赌注乘以下注类型的赔率
    public int Payout => Wager * Type.Payout;
}

```