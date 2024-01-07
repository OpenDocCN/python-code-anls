# `basic-computer-games\16_Bug\csharp\Parts\IPart.cs`

```

# 声明一个命名空间 BugGame.Parts，用于组织和管理相关的类和接口
namespace BugGame.Parts;

# 声明一个接口 IPart，表示游戏部件的接口
internal interface IPart
{
    # 声明一个只读属性 Name，用于获取部件的名称
    string Name { get; }
}

```