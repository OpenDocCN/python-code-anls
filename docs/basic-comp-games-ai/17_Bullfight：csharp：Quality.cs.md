# `17_Bullfight\csharp\Quality.cs`

```
// 声明一个枚举类型，用于列举游戏中不同的质量级别
namespace Game
{
    /// <summary>
    /// Enumerates the different levels of quality in the game.
    /// </summary>
    /// <remarks>
    /// Quality applies both to the bull and to the help received from the
    /// toreadores and picadores.  Note that the ordinal values are significant
    /// (these are used in various calculations).
    /// </remarks>
    public enum Quality
    {
        Superb  = 1, // 代表最高质量级别
        Good    = 2, // 代表较高质量级别
        Fair    = 3, // 代表一般质量级别
        Poor    = 4, // 代表较低质量级别
        Awful   = 5  // 代表最低质量级别
    }
}
```