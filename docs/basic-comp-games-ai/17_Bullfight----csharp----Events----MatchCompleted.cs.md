# `basic-computer-games\17_Bullfight\csharp\Events\MatchCompleted.cs`

```

// 在 Game.Events 命名空间下定义了一个记录类型 MatchCompleted，表示战斗已经完成
// 该记录类型继承自 Event 类
public sealed record MatchCompleted(ActionResult Result, bool ExtremeBravery, Reward Reward) : Event;
// MatchCompleted 类型包含三个属性：Result（表示战斗结果）、ExtremeBravery（表示是否极度勇敢）、Reward（表示奖励）

```