# `basic-computer-games\18_Bullseye\csharp\Player.cs`

```
namespace Bullseye
{
    /// <summary>
    /// Object to track the name and score of a player
    /// </summary>
    public class Player
    {
        /// <summary>
        /// Creates a play with the given name
        /// </summary>
        /// <param name="name">Name of the player</param>
        public Player(string name)
        {
            // 设置玩家的名字
            Name = name;
            // 初始化玩家的分数为0
            Score = 0;
        }

        /// <summary>
        /// Name of the player
        /// </summary>
        public string Name { get; private set; }

        /// <summary>
        /// Current score of the player
        /// </summary>
        public int Score { get; set; }
    }
}
```