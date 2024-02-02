# `basic-computer-games\60_Mastermind\csharp\ColorInfo.cs`

```py
// 引入 System 命名空间
using System;

// 声明 Game 命名空间
namespace Game
{
    /// <summary>
    /// 存储有关颜色的信息。
    /// </summary>
    // 声明 ColorInfo 记录
    public record ColorInfo
    {
        /// <summary>
        /// 获取表示颜色的单个字符。
        /// </summary>
        // 声明 ShortName 属性，用于获取颜色的简称
        public char ShortName { get; init; }

        /// <summary>
        /// 获取颜色的全名。
        /// </summary>
        // 声明 LongName 属性，用于获取颜色的全名，默认值为空字符串
        public string LongName { get; init; } = String.Empty;
    }
}
```