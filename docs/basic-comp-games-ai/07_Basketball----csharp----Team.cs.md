# `basic-computer-games\07_Basketball\csharp\Team.cs`

```py
# 使用篮球战术库中的 Plays 命名空间
using Basketball.Plays;

# 声明篮球命名空间
namespace Basketball;

# 声明内部记录类型 Team，包含球队名称和战术解析器
internal record Team(string Name, Play PlayResolver)
{
    # 重写 ToString 方法，返回球队名称
    public override string ToString() => Name;

    # 解析战术，传入比分板对象，调用 PlayResolver 解析方法
    public bool ResolvePlay(Scoreboard scoreboard) => PlayResolver.Resolve(scoreboard);
}
```