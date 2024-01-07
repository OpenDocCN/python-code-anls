# `basic-computer-games\17_Bullfight\csharp\Quality.cs`

```

// 命名空间 Game，定义了游戏中不同质量级别的枚举类型
namespace Game
{
    /// <summary>
    /// Enumerates the different levels of quality in the game.
    /// 枚举了游戏中不同的质量级别。
    /// </summary>
    /// <remarks>
    /// Quality applies both to the bull and to the help received from the
    /// toreadores and picadores.  Note that the ordinal values are significant
    /// (these are used in various calculations).
    /// 质量既适用于公牛，也适用于来自斗牛士和刺激者的帮助。请注意，序数值是重要的（这些值在各种计算中使用）。
    /// </remarks>
    public enum Quality
    {
        Superb  = 1, // 一级质量
        Good    = 2, // 二级质量
        Fair    = 3, // 三级质量
        Poor    = 4, // 四级质量
        Awful   = 5  // 五级质量
    }
}

```