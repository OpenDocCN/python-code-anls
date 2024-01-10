# `basic-computer-games\08_Batnum\csharp\BatnumGame.cs`

```
// 引用命名空间
using Batnum.Properties;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// 命名空间定义
namespace Batnum
{
    // 枚举定义，表示游戏的胜利选项
    public enum WinOptions
    {
        /// <summary>
        /// 最后一个玩家赢
        /// </summary>
        WinWithTakeLast = 1,
        /// <summary>
        /// 避免成为最后一个玩家
        /// </summary>
        WinWithAvoidLast = 2
    }

    // 枚举定义，表示玩家类型
    public enum Players
    {
        Computer = 1,
        Human = 2
    }

    // 游戏类定义
    public class BatnumGame
    {
    }
}
```