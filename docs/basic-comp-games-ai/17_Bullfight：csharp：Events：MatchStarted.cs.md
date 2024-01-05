# `d:/src/tocomm/basic-computer-games\17_Bullfight\csharp\Events\MatchStarted.cs`

```
// 命名空间声明，指定了该类所属的命名空间
namespace Game.Events
{
    /// <summary>
    /// 表示一场新比赛已经开始的事件
    /// </summary>
    // 使用 record 关键字声明一个记录类型，记录类型是不可变的数据类型
    public sealed record MatchStarted(
        // 声明了 BullQuality 属性，表示斗牛的质量
        Quality BullQuality,
        // 声明了 ToreadorePerformance 属性，表示斗牛士的表现质量
        Quality ToreadorePerformance,
        // 声明了 PicadorePerformance 属性，表示刺马士的表现质量
        Quality PicadorePerformance,
        // 声明了 ToreadoresKilled 属性，表示被杀死的斗牛士数量
        int ToreadoresKilled,
        // 声明了 PicadoresKilled 属性，表示被杀死的刺马士数量
        int PicadoresKilled,
        // 声明了 HorsesKilled 属性，表示被杀死的马匹数量
        int HorsesKilled) : Event;
}
```