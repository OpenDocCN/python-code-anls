# `basic-computer-games\71_Poker\csharp\Strategies\Bet.cs`

```

# 命名空间为Poker.Strategies，表示该类属于Poker命名空间下的Strategies子命名空间
internal class Bet : Strategy
# 创建一个名为Bet的类，该类继承自Strategy类，并且只能在当前程序集内部访问
{
    public Bet(int amount) => Value = amount;
    # 创建一个公共的构造函数，接受一个整数参数amount，并将其赋值给Value属性

    public override int Value { get; }
    # 创建一个公共的重写属性Value，表示下注的数额
}

```