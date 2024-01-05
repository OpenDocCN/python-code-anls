# `16_Bug\csharp\Parts\Leg.cs`

```
# 命名空间 BugGame.Parts
namespace BugGame.Parts;

# 内部类 Leg 实现接口 IPart
internal class Leg : IPart
{
    # 公共属性 Name 返回 Leg 类的名称
    public string Name => nameof(Leg);
}
```