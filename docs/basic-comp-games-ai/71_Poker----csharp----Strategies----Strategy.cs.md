# `basic-computer-games\71_Poker\csharp\Strategies\Strategy.cs`

```
# 定义一个名为 Strategy 的命名空间，用于存放不同的扑克策略类
internal abstract class Strategy
{
    # 定义一个静态的 None 策略对象
    public static Strategy None = new None();
    # 定义一个静态的 Fold 策略对象
    public static Strategy Fold = new Fold();
    # 定义一个静态的 Check 策略对象
    public static Strategy Check = new Check();
    # 定义一个静态的 Raise 策略对象
    public static Strategy Raise = new Raise();
    # 定义一个静态的 Bet 策略对象，接受浮点数类型的下注金额
    public static Strategy Bet(float amount) => new Bet((int)amount);
    # 定义一个静态的 Bet 策略对象，接受整数类型的下注金额
    public static Strategy Bet(int amount) => new Bet(amount);
    # 定义一个静态的 Bluff 策略对象，接受整数类型的下注金额和可选的保留掩码
    public static Strategy Bluff(int amount, int? keepMask = null) => new Bluff(amount, keepMask);

    # 定义一个抽象的 Value 属性，用于表示策略的价值
    public abstract int Value { get; }
    # 定义一个虚拟的 KeepMask 属性，用于表示保留掩码
    public virtual int? KeepMask { get; }
}
```