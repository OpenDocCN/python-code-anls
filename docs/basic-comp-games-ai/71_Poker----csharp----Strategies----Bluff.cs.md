# `basic-computer-games\71_Poker\csharp\Strategies\Bluff.cs`

```

# 命名空间为Poker.Strategies，表示这个类属于Poker.Strategies命名空间
internal class Bluff : Bet
{
    # 构造函数，接受一个amount参数和一个可空的keepMask参数，调用基类的构造函数
    public Bluff(int amount, int? keepMask)
        : base(amount)
    {
        # 设置当前类的KeepMask属性为传入的keepMask参数
        KeepMask = keepMask;
    }

    # 重写基类的KeepMask属性，表示当前类的保留掩码
    public override int? KeepMask { get; }
}

```