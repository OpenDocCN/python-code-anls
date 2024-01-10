# `basic-computer-games\17_Bullfight\csharp\Events\MatchStarted.cs`

```
// 声明在 Game.Events 命名空间下的 MatchStarted 事件记录，表示一场新比赛开始
namespace Game.Events
{
    /// <summary>
    /// 表示一场新比赛开始的事件记录
    /// </summary>
    public sealed record MatchStarted(
        // 母牛质量
        Quality BullQuality,
        // 斗牛士表现质量
        Quality ToreadorePerformance,
        // 骑士表现质量
        Quality PicadorePerformance,
        // 被杀的斗牛士数量
        int ToreadoresKilled,
        // 被杀的骑士数量
        int PicadoresKilled,
        // 被杀的马匹数量
        int HorsesKilled) : Event;
}
```