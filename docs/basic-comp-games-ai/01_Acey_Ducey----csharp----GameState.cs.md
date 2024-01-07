# `basic-computer-games\01_Acey_Ducey\csharp\GameState.cs`

```

// 引入命名空间
using System;
using System.Collections.Generic;
using System.Text;

// 命名空间 AceyDucey
namespace AceyDucey
{
    /// <summary>
    /// GameState 类用于跟踪游戏进行中的所有游戏变量
    /// </summary>
    internal class GameState
    {
        /// <summary>
        /// 玩家当前拥有多少钱？
        /// </summary>
        internal int Money { get; set; }

        /// <summary>
        /// 游戏中玩家曾经拥有的最高金额是多少？
        /// </summary>
        internal int MaxMoney { get; set; }

        /// <summary>
        /// 玩了多少轮？
        /// </summary>
        internal int TurnCount { get; set; }

        /// <summary>
        /// 类构造函数 -- 将所有值初始化为它们的默认值。
        /// </summary>
        internal GameState()
        {
            // 将 Money 设置为 100，给玩家起始余额。更改此值将改变玩家的初始金额。
            Money = 100;
            MaxMoney = Money;
            TurnCount = 0;
        }
    }
}

```