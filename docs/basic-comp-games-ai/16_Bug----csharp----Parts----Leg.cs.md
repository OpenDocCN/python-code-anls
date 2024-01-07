# `basic-computer-games\16_Bug\csharp\Parts\Leg.cs`

```

# 声明一个名为 BugGame.Parts 的命名空间，用于组织和管理相关类
namespace BugGame.Parts;

# 声明一个名为 Leg 的类，实现了 IPart 接口，表示它是一个部件
internal class Leg : IPart
{
    # 声明一个公共的只读属性 Name，返回 Leg 类的名称
    public string Name => nameof(Leg);
}

```