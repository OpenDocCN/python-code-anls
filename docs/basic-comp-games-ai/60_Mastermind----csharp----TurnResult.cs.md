# `basic-computer-games\60_Mastermind\csharp\TurnResult.cs`

```py
// 命名空间 Game，存储了玩家回合的结果
namespace Game
{
    /// <summary>
    /// 存储了玩家回合的结果
    /// </summary>
    public record TurnResult
    {
        /// <summary>
        /// 获取玩家猜测的代码
        /// </summary>
        public Code Guess { get; }

        /// <summary>
        /// 获取猜测结果中黑色提示棒的数量
        /// </summary>
        public int Blacks { get; }

        /// <summary>
        /// 获取猜测结果中白色提示棒的数量
        /// </summary>
        public int Whites { get; }

        /// <summary>
        /// 初始化 TurnResult 记录的新实例
        /// </summary>
        /// <param name="guess">
        /// 玩家的猜测
        /// </param>
        /// <param name="blacks">
        /// 黑色提示棒的数量
        /// </param>
        /// <param name="whites">
        /// 白色提示棒的数量
        /// </param>
        public TurnResult(Code guess, int blacks, int whites) =>
            (Guess, Blacks, Whites) = (guess, blacks, whites);
    }
}
```