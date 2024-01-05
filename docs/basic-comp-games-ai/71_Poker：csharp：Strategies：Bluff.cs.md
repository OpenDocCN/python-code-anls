# `d:/src/tocomm/basic-computer-games\71_Poker\csharp\Strategies\Bluff.cs`

```
namespace Poker.Strategies; // 命名空间声明，表示该类所属的命名空间

internal class Bluff : Bet // 声明一个名为Bluff的类，继承自Bet类，使用internal关键字表示只能在当前程序集内访问
{
    public Bluff(int amount, int? keepMask) // 声明一个名为Bluff的构造函数，接受amount和keepMask两个参数
        : base(amount) // 调用基类Bet的构造函数，传入amount参数
    {
        KeepMask = keepMask; // 将传入的keepMask赋值给当前类的KeepMask属性
    }

    public override int? KeepMask { get; } // 声明一个名为KeepMask的属性，表示该类的保留掩码，使用override关键字表示重写基类的属性
}
```