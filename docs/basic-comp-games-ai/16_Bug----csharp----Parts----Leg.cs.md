# `basic-computer-games\16_Bug\csharp\Parts\Leg.cs`

```py
# 声明 BugGame.Parts 命名空间下的 Leg 类，并实现 IPart 接口
internal class Leg : IPart
{
    # 声明 Name 属性，返回 Leg 类的名称
    public string Name => nameof(Leg);
}
```