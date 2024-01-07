# `basic-computer-games\17_Bullfight\csharp\ActionResult.cs`

```

// 引入所需的命名空间
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// 命名空间 Game
namespace Game
{
    /// <summary>
    /// Enumerates the different possible outcomes of the player's action.
    /// 枚举了玩家行动的不同可能结果。
    /// </summary>
    public enum ActionResult
    {
        /// <summary>
        /// The fight continues.
        /// 战斗继续。
        /// </summary>
        FightContinues,

        /// <summary>
        /// The player fled from the ring.
        /// 玩家逃离了斗牛场。
        /// </summary>
        PlayerFlees,

        /// <summary>
        /// The bull has gored the player.
        /// 公牛刺伤了玩家。
        /// </summary>
        BullGoresPlayer,

        /// <summary>
        /// The bull killed the player.
        /// 公牛杀死了玩家。
        /// </summary>
        BullKillsPlayer,

        /// <summary>
        /// The player killed the bull.
        /// 玩家杀死了公牛。
        /// </summary>
        PlayerKillsBull,

        /// <summary>
        /// The player attempted to kill the bull and both survived.
        /// 玩家试图杀死公牛，但双方都幸存。
        /// </summary>
        Draw
    }
}

```