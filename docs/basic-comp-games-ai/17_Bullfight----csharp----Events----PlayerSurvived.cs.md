# `basic-computer-games\17_Bullfight\csharp\Events\PlayerSurvived.cs`

```

// 声明一个命名空间 Game.Events，用于组织和管理事件相关的类
namespace Game.Events
{
    /// <summary>
    /// 表示玩家成功躲过公牛的攻击的事件
    /// </summary>
    // 声明一个不可变的记录类型 PlayerSurvived，继承自 Event 类
    public sealed record PlayerSurvived() : Event;
}

```