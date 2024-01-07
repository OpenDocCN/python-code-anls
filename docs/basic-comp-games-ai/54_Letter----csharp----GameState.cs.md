# `basic-computer-games\54_Letter\csharp\GameState.cs`

```

// 命名空间 Letter
namespace Letter
{
    /// <summary>
    /// Holds the current state.（保存当前状态）
    /// </summary>
    internal class GameState
    {
        /// <summary>
        /// Initialise the game state with a random letter.（用随机字母初始化游戏状态）
        /// </summary>
        public GameState()
        {
            Letter = GetRandomLetter(); // 使用 GetRandomLetter 方法获取随机字母
            GuessesSoFar = 0; // 猜测次数初始化为 0
        }

        /// <summary>
        /// The letter that the user is guessing.（用户正在猜测的字母）
        /// </summary>
        public char Letter { get; set; } // 字母属性

        /// <summary>
        /// The number of guesses the user has had so far.（用户到目前为止的猜测次数）
        /// </summary>
        public int GuessesSoFar { get; set; } // 猜测次数属性

        /// <summary>
        /// Get a random character (A-Z) for the user to guess.（获取一个用户要猜测的随机字符（A-Z））
        /// </summary>
        internal static char GetRandomLetter() // 静态方法，用于获取随机字母
        {
            var random = new Random(); // 创建 Random 对象
            var randomNumber = random.Next(0, 26); // 生成 0 到 25 之间的随机数
            return (char)('A' + randomNumber); // 返回对应的随机字母
        }
    }
}

```