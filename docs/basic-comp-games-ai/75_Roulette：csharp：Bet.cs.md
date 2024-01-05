# `75_Roulette\csharp\Bet.cs`

```
namespace Roulette;  # 命名空间声明，用于组织和管理代码

internal record struct Bet(BetType Type, int Number, int Wager)  # 定义一个名为Bet的记录结构，包含BetType类型的Type、整数类型的Number和Wager字段
{
    public int Payout => Wager * Type.Payout;  # 定义一个公共的只读属性Payout，返回Wager乘以Type.Payout的结果
}
```