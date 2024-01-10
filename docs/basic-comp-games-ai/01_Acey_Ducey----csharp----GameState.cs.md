# `basic-computer-games\01_Acey_Ducey\csharp\GameState.cs`

```
namespace AceyDucey
{
    /// <summary>
    /// The GameState class keeps track of all the game variables while the game is being played
    /// 游戏状态类用于跟踪游戏进行中的所有游戏变量
    /// </summary>
    internal class GameState
    {

        /// <summary>
        /// How much money does the player have at the moment?
        /// 玩家目前有多少钱？
        /// </summary>
        internal int Money { get; set; }

        /// <summary>
        /// What's the highest amount of money they had at any point in the game?
        /// 玩家在游戏中的任何时刻拥有的最高金额是多少？
        /// </summary>
        internal int MaxMoney { get; set; }

        /// <summary>
        /// How many turns have they played?
        /// 他们玩了多少轮？
        /// </summary>
        internal int TurnCount { get; set; }

        /// <summary>
        /// Class constructor -- initialise all values to their defaults.
        /// 类构造函数 - 将所有值初始化为它们的默认值。
        /// </summary>
        internal GameState()
        {
            // Setting Money to 100 gives the player their starting balance. Changing this will alter how much they have to begin with.
            // 将 Money 设置为 100 可以给玩家他们的起始余额。更改这个值将改变他们最初拥有的金额。
            Money = 100;
            MaxMoney = Money;
            TurnCount = 0;
        }
    }
}
```