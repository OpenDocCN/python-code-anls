# `basic-computer-games\60_Mastermind\csharp\TurnResult.cs`

```

// 命名空间 Game
namespace Game
{
    /// <summary>
    /// 存储玩家回合的结果
    /// </summary>
    public record TurnResult
    {
        /// <summary>
        /// 获取玩家猜测的代码
        /// </summary>
        public Code Guess { get; }

        /// <summary>
        /// 获取猜测结果中黑色标志的数量
        /// </summary>
        public int Blacks { get; }

        /// <summary>
        /// 获取猜测结果中白色标志的数量
        /// </summary>
        public int Whites { get; }

        /// <summary>
        /// 初始化 TurnResult 记录的新实例
        /// </summary>
        /// <param name="guess">
        /// 玩家的猜测
        /// </param>
        /// <param name="blacks">
        /// 黑色标志的数量
        /// </param>
        /// <param name="whites">
        /// 白色标志的数量
        /// </param>
        public TurnResult(Code guess, int blacks, int whites) =>
            (Guess, Blacks, Whites) = (guess, blacks, whites);
    }
}

```