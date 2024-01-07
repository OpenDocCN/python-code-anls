# `basic-computer-games\17_Bullfight\csharp\Events\MatchStarted.cs`

```

// 在 Game.Events 命名空间下定义了一个新的事件类型 MatchStarted
namespace Game.Events
{
    /// <summary>
    /// 表示一场新比赛已经开始
    /// </summary>
    // 使用 record 关键字定义了一个不可变的数据记录类型 MatchStarted，继承自 Event 类型
    public sealed record MatchStarted(
        // 表示斗牛的质量
        Quality BullQuality,
        // 表示斗牛士的表现质量
        Quality ToreadorePerformance,
        // 表示骑士的表现质量
        Quality PicadorePerformance,
        // 表示被杀死的斗牛士数量
        int ToreadoresKilled,
        // 表示被杀死的骑士数量
        int PicadoresKilled,
        // 表示被杀死的马匹数量
        int HorsesKilled) : Event;
}

```