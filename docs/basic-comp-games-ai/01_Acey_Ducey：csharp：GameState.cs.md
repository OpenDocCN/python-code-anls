# `01_Acey_Ducey\csharp\GameState.cs`

```
        internal int HighestMoney { get; set; }

        /// <summary>
        /// How many rounds have been played so far?
        /// </summary>
        internal int RoundsPlayed { get; set; }

        /// <summary>
        /// Constructor for the GameState class
        /// </summary>
        internal GameState()
        {
            // Initialize the game state variables
            Money = 100; // Set the initial amount of money to 100
            HighestMoney = 100; // Set the initial highest amount of money to 100
            RoundsPlayed = 0; // Set the initial rounds played to 0
        }
    }
}
        internal int MaxMoney { get; set; }  // 定义一个内部属性 MaxMoney，用于存储最大金额

        /// <summary>
        /// How many turns have they played?  // 描述了 TurnCount 属性的作用，用于存储玩家已经进行的回合数
        /// </summary>
        internal int TurnCount { get; set; }  // 定义一个内部属性 TurnCount，用于存储回合数

        /// <summary>
        /// Class constructor -- initialise all values to their defaults.  // 描述了 GameState 类的构造函数，用于初始化所有值为默认值
        /// </summary>
        internal GameState()
        {
            // Setting Money to 100 gives the player their starting balance. Changing this will alter how much they have to begin with.
            Money = 100;  // 将 Money 属性初始化为 100，给玩家一个起始金额
            MaxMoney = Money;  // 将 MaxMoney 属性初始化为 Money 的值
            TurnCount = 0;  // 将 TurnCount 属性初始化为 0
        }
    }
}
```