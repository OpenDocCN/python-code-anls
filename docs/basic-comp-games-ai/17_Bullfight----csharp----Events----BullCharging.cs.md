# `basic-computer-games\17_Bullfight\csharp\Events\BullCharging.cs`

```py
// 在 Game.Events 命名空间下定义了一个事件类 BullCharging，表示公牛正在冲向玩家
// 使用 sealed 关键字标记该类为密封类，不可被继承
// 使用 record 关键字定义记录类型，包含一个 PassNumber 属性
// 继承自 Event 类
public sealed record BullCharging(int PassNumber) : Event;
```