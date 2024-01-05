# `54_Letter\csharp\GameState.cs`

```
        /// <summary>
        /// The number of guesses made so far.
        /// </summary>
        public int GuessesSoFar { get; set; }

        /// <summary>
        /// Get a random letter from A to Z.
        /// </summary>
        /// <returns>A random letter</returns>
        private char GetRandomLetter()
        {
            // Generate a random number between 65 and 90, which corresponds to the ASCII values of A to Z
            int randomNum = new Random().Next(65, 91);
            // Convert the random number to a character and return it
            return (char)randomNum;
        }
    }
}
/// <summary>
/// 用户迄今为止猜测的次数。
/// </summary>
public int GuessesSoFar { get; set; }

/// <summary>
/// 获取一个随机字符（A-Z），供用户猜测。
/// </summary>
internal static char GetRandomLetter()
{
    // 创建一个随机数生成器对象
    var random = new Random();
    // 生成一个0到25之间的随机数
    var randomNumber = random.Next(0, 26);
    // 将随机数转换为对应的字符（A-Z）
    return (char)('A' + randomNumber);
}
```