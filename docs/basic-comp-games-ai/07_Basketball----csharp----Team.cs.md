# `basic-computer-games\07_Basketball\csharp\Team.cs`

```

# 使用篮球战术库中的 Plays 命名空间
using Basketball.Plays;

# 声明一个命名空间 Basketball
namespace Basketball;

# 声明一个内部的记录类型 Team，包含球队名称和战术解析器
internal record Team(string Name, Play PlayResolver)
{
    # 重写 ToString 方法，返回球队名称
    public override string ToString() => Name;

    # 声明一个方法 ResolvePlay，接受一个 Scoreboard 对象，调用 PlayResolver 解析战术
    public bool ResolvePlay(Scoreboard scoreboard) => PlayResolver.Resolve(scoreboard);
}

```