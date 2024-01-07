# `basic-computer-games\71_Poker\csharp\Strategies\Strategy.cs`

```

// 命名空间为Poker.Strategies，表示该类属于Poker.Strategies命名空间
internal abstract class Strategy
{
    // 定义静态字段None，表示没有任何策略
    public static Strategy None = new None();
    // 定义静态字段Fold，表示放弃策略
    public static Strategy Fold = new Fold();
    // 定义静态字段Check，表示检查策略
    public static Strategy Check = new Check();
    // 定义静态字段Raise，表示加注策略
    public static Strategy Raise = new Raise();
    // 定义静态方法Bet，表示下注策略，参数为浮点数金额，返回下注策略对象
    public static Strategy Bet(float amount) => new Bet((int)amount);
    // 定义静态方法Bet，表示下注策略，参数为整数金额，返回下注策略对象
    public static Strategy Bet(int amount) => new Bet(amount);
    // 定义静态方法Bluff，表示虚张声势策略，参数为金额和保留掩码，返回虚张声势策略对象
    public static Strategy Bluff(int amount, int? keepMask = null) => new Bluff(amount, keepMask);

    // 抽象属性Value，表示策略的价值
    public abstract int Value { get; }
    // 虚属性KeepMask，表示保留掩码
    public virtual int? KeepMask { get; }
}

```