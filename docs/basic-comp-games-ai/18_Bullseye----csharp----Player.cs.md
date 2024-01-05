# `18_Bullseye\csharp\Player.cs`

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
        public string Name { get; private set; }  // 声明一个公共的字符串类型的属性 Name，只能在类的内部进行设置

        /// <summary>
        /// Current score of the player
        /// </summary>
        public int Score { get; set; }  // 声明一个公共的整数类型的属性 Score，可以在类的内外进行设置
    }
}
```