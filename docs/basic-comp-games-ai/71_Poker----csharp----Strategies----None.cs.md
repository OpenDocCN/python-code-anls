# `basic-computer-games\71_Poker\csharp\Strategies\None.cs`

```py
# 创建一个名为None的内部类，继承自Strategy类
internal class None : Strategy
{
    # 覆盖父类的Value属性，返回-1
    public override int Value => -1;
}
```