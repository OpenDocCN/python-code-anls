# `basic-computer-games\60_Mastermind\csharp\ColorInfo.cs`

```

// 命名空间 Game，存储关于颜色的信息
namespace Game
{
    /// <summary>
    /// 存储关于颜色的信息
    /// </summary>
    public record ColorInfo
    {
        /// <summary>
        /// 获取代表颜色的单个字符
        /// </summary>
        public char ShortName { get; init; }

        /// <summary>
        /// 获取颜色的全名
        /// </summary>
        public string LongName { get; init; } = String.Empty;
    }
}

```