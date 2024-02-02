# `basic-computer-games\17_Bullfight\csharp\ActionResult.cs`

```py
// 命名空间 Game 包含了游戏相关的类和枚举
namespace Game
{
    /// <summary>
    /// 枚举了玩家行动的不同可能结果
    /// </summary>
    public enum ActionResult
    {
        /// <summary>
        /// 战斗继续
        /// </summary>
        FightContinues,

        /// <summary>
        /// 玩家逃离了战斗
        /// </summary>
        PlayerFlees,

        /// <summary>
        /// 公牛刺伤了玩家
        /// </summary>
        BullGoresPlayer,

        /// <summary>
        /// 公牛杀死了玩家
        /// </summary>
        BullKillsPlayer,

        /// <summary>
        /// 玩家杀死了公牛
        /// </summary>
        PlayerKillsBull,

        /// <summary>
        /// 玩家试图杀死公牛，但双方都幸存
        /// </summary>
        Draw
    }
}
```