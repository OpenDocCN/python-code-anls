# `17_Bullfight\csharp\Events\BullCharging.cs`

```
// 在 Game.Events 命名空间下定义了一个记录类型 BullCharging，表示公牛正在冲向玩家
// 该记录类型继承自 Event 类
public sealed record BullCharging(int PassNumber) : Event;
```