# `71_Poker\csharp\Strategies\Bet.cs`

```
# 命名空间为Poker.Strategies
namespace Poker.Strategies;

# 内部类Bet继承自Strategy
internal class Bet : Strategy
{
    # 构造函数，接受一个整数参数amount，并将其赋值给Value属性
    public Bet(int amount) => Value = amount;

    # 重写父类的Value属性，只有get访问器
    public override int Value { get; }
}
```