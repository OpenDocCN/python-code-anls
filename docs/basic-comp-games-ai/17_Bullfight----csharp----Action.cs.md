# `basic-computer-games\17_Bullfight\csharp\Action.cs`

```

// 命名空间 Game，定义了游戏中玩家在每轮战斗中可以采取的不同动作
namespace Game
{
    /// <summary>
    /// 枚举了玩家在每轮战斗中可以采取的不同动作
    /// </summary>
    public enum Action
    {
        /// <summary>
        /// 躲避公牛
        /// </summary>
        Dodge,

        /// <summary>
        /// 击杀公牛
        /// </summary>
        Kill,

        /// <summary>
        /// 固定在原地，不采取任何行动
        /// </summary>
        Panic
    }
}

```