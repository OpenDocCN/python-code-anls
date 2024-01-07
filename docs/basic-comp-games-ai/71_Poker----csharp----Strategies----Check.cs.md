# `basic-computer-games\71_Poker\csharp\Strategies\Check.cs`

```

# 命名空间为Poker.Strategies，表示该类属于Poker命名空间下的Strategies子命名空间
internal class Check : Strategy
# Check类继承自Strategy类，表示Check类是Strategy类的子类，并且只能在当前程序集内部访问
{
    public override int Value => 0;
    # Value属性的重写，返回值为0，表示Check策略的值为0
}

```