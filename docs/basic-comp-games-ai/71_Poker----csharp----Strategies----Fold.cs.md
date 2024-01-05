# `71_Poker\csharp\Strategies\Fold.cs`

```
namespace Poker.Strategies; // 声明命名空间为Poker.Strategies

internal class Fold : Strategy // 声明一个内部类Fold，继承自Strategy类
{
    public override int Value => -1; // 重写父类的Value属性，返回-1
}
```