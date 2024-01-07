# `basic-computer-games\71_Poker\csharp\Strategies\Fold.cs`

```

# 命名空间为Poker.Strategies，表示该类属于Poker命名空间下的Strategies子命名空间
internal class Fold : Strategy
{
    # 重写父类的Value属性，返回-1，表示该策略的价值为-1
    public override int Value => -1;
}

```