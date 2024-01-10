# `basic-computer-games\16_Bug\csharp\Parts\Feeler.cs`

```
# 定义了一个名为 Feeler 的内部类，实现了 IPart 接口
internal class Feeler : IPart
{
    # 实现了 IPart 接口中的 Name 属性，返回 Feeler 的名称
    public string Name => nameof(Feeler);
}
```