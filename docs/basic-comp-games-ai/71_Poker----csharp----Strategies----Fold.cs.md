# `basic-computer-games\71_Poker\csharp\Strategies\Fold.cs`

```
# 创建一个名为Fold的内部类，继承自Strategy类
internal class Fold : Strategy
{
    # 重写Value属性，返回-1，表示放弃策略的价值
    public override int Value => -1;
}
```