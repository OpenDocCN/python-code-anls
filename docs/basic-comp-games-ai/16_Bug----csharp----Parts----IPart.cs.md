# `basic-computer-games\16_Bug\csharp\Parts\IPart.cs`

```
# 命名空间 BugGame.Parts 下的接口 IPart
# 接口定义了一个属性 Name，用于获取部件的名称
namespace BugGame.Parts;

# 定义了一个内部接口 IPart
internal interface IPart
{
    # 获取部件的名称
    string Name { get; }
}
```