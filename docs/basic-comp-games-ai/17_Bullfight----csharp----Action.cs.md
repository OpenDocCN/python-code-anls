# `basic-computer-games\17_Bullfight\csharp\Action.cs`

```
// 命名空间 Game，定义游戏中玩家在每一轮战斗中可以采取的不同动作
namespace Game
{
    /// <summary>
    /// Enumerates the different actions that the player can take on each round
    /// of the fight.
    /// </summary>
    // 定义枚举类型 Action，列举玩家在每一轮战斗中可以采取的不同动作
    public enum Action
    {
        /// <summary>
        /// Dodge the bull.
        /// </summary>
        // 躲避公牛
        Dodge,

        /// <summary>
        /// Kill the bull.
        /// </summary>
        // 击杀公牛
        Kill,

        /// <summary>
        /// Freeze in place and don't do anything.
        /// </summary>
        // 原地冻结，不做任何动作
        Panic
    }
}
```