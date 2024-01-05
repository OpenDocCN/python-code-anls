# `07_Basketball\csharp\Team.cs`

```
using Basketball.Plays;  # 导入 Basketball.Plays 模块

namespace Basketball;  # 声明命名空间 Basketball

internal record Team(string Name, Play PlayResolver)  # 声明一个内部记录类型 Team，包含名称和 PlayResolver 属性
{
    public override string ToString() => Name;  # 重写 ToString 方法，返回球队名称

    public bool ResolvePlay(Scoreboard scoreboard) => PlayResolver.Resolve(scoreboard);  # 定义 ResolvePlay 方法，使用 PlayResolver 解析比赛情况
}
```