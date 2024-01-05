# `d:/src/tocomm/basic-computer-games\17_Bullfight\csharp\Events\PlayerSurvived.cs`

```
// 在 Game.Events 命名空间下定义了一个事件记录，表示玩家成功躲过了公牛的攻击
// 该记录是一个不可变的数据结构，继承自 Event 类
public sealed record PlayerSurvived() : Event;
```