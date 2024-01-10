# `basic-computer-games\17_Bullfight\csharp\Events\MatchCompleted.cs`

```
// 声明在游戏事件命名空间下的比赛完成事件
namespace Game.Events
{
    /// <summary>
    /// 表示比赛已经完成
    /// </summary>
    // 声明一个不可变的记录类型，包含比赛结果、是否极度勇敢和奖励
    public sealed record MatchCompleted(ActionResult Result, bool ExtremeBravery, Reward Reward) : Event;
}
```