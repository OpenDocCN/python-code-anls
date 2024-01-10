# `basic-computer-games\54_Letter\csharp\GameState.cs`

```
// 声明 Letter 命名空间
namespace Letter
{
    /// <summary>
    /// Holds the current state.（保存当前状态）
    /// </summary>
    // 声明内部类 GameState
    internal class GameState
    {
        /// <summary>
        /// Initialise the game state with a random letter.（用随机字母初始化游戏状态）
        /// </summary>
        // 构造函数，初始化游戏状态，包括随机字母和猜测次数
        public GameState()
        {
            Letter = GetRandomLetter();
            GuessesSoFar = 0;
        }

        /// <summary>
        /// The letter that the user is guessing.（用户正在猜测的字母）
        /// </summary>
        // 公共属性，用于存储用户猜测的字母
        public char Letter { get; set; }

        /// <summary>
        /// The number of guesses the user has had so far.（用户到目前为止的猜测次数）
        /// </summary>
        // 公共属性，用于存储用户到目前为止的猜测次数
        public int GuessesSoFar { get; set; }

        /// <summary>
        /// Get a random character (A-Z) for the user to guess.（获取一个用户要猜测的随机字符（A-Z））
        /// </summary>
        // 静态方法，用于获取一个随机字符（A-Z）供用户猜测
        internal static char GetRandomLetter()
        {
            var random = new Random();
            var randomNumber = random.Next(0, 26);
            return (char)('A' + randomNumber);
        }
    }
}
```