# `basic-computer-games\17_Bullfight\csharp\Events\PlayerGored.cs`

```
// 声明在游戏事件命名空间下的玩家被公牛顶到的事件
namespace Game.Events
{
    /// <summary>
    /// 表示玩家被公牛顶到的事件
    /// </summary>
    // 声明一个不可变的记录类型，记录玩家是否处于恐慌状态和是否是第一次被顶到
    public sealed record PlayerGored(bool Panicked, bool FirstGoring) : Event;
}
```