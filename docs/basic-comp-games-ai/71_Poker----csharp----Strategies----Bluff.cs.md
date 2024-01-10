# `basic-computer-games\71_Poker\csharp\Strategies\Bluff.cs`

```
# 命名空间为Poker.Strategies，表示该类属于Poker.Strategies命名空间
internal class Bluff : Bet
{
    # 构造函数，接受赌注金额和保留掩码作为参数
    public Bluff(int amount, int? keepMask)
        : base(amount)  # 调用基类Bet的构造函数，传入赌注金额
    {
        KeepMask = keepMask;  # 设置保留掩码
    }

    # 重写基类的KeepMask属性，表示保留掩码
    public override int? KeepMask { get; }
}
```