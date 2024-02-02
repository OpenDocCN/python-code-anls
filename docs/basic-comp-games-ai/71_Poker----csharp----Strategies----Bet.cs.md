# `basic-computer-games\71_Poker\csharp\Strategies\Bet.cs`

```py
# 命名空间为Poker.Strategies，表示该类属于Poker.Strategies命名空间
internal class Bet : Strategy
{
    # 构造函数，接受一个整数参数amount，并将其赋值给Value属性
    public Bet(int amount) => Value = amount;

    # 重写父类的Value属性，表示下注的数额
    public override int Value { get; }
}
```