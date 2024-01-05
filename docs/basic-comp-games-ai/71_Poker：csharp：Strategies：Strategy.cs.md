# `71_Poker\csharp\Strategies\Strategy.cs`

```
namespace Poker.Strategies;  // 命名空间声明，表示该类属于Poker.Strategies命名空间

internal abstract class Strategy  // 声明一个内部的抽象类Strategy
{
    public static Strategy None = new None();  // 声明一个静态的Strategy对象None，并初始化为None类的实例
    public static Strategy Fold = new Fold();  // 声明一个静态的Strategy对象Fold，并初始化为Fold类的实例
    public static Strategy Check = new Check();  // 声明一个静态的Strategy对象Check，并初始化为Check类的实例
    public static Strategy Raise = new Raise();  // 声明一个静态的Strategy对象Raise，并初始化为Raise类的实例
    public static Strategy Bet(float amount) => new Bet((int)amount);  // 声明一个静态的Strategy对象Bet，接受一个浮点数参数amount，并返回一个Bet类的实例
    public static Strategy Bet(int amount) => new Bet(amount);  // 声明一个静态的Strategy对象Bet，接受一个整数参数amount，并返回一个Bet类的实例
    public static Strategy Bluff(int amount, int? keepMask = null) => new Bluff(amount, keepMask);  // 声明一个静态的Strategy对象Bluff，接受一个整数参数amount和一个可空的整数参数keepMask，并返回一个Bluff类的实例

    public abstract int Value { get; }  // 声明一个抽象的整数属性Value，表示策略的价值
    public virtual int? KeepMask { get; }  // 声明一个虚拟的可空整数属性KeepMask，表示保留的牌的掩码
}
```