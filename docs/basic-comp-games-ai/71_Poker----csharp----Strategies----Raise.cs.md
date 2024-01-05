# `71_Poker\csharp\Strategies\Raise.cs`

```
namespace Poker.Strategies;  // 声明命名空间为Poker.Strategies，用于组织和管理类和其他类型

internal class Raise : Bet  // 声明一个内部类Raise，继承自Bet类
{
    public Raise() : base(2) { }  // 声明Raise类的构造函数，调用基类Bet的构造函数并传入参数2
}
```