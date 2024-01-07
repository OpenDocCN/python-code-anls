# `basic-computer-games\71_Poker\csharp\Strategies\None.cs`

```

# 命名空间为Poker.Strategies，表示该类属于Poker.Strategies命名空间
internal class None : Strategy
{
    # 覆盖父类的Value属性，返回-1，表示该策略无效
    public override int Value => -1;
}

```