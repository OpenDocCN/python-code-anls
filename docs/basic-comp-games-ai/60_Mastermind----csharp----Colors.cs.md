# `basic-computer-games\60_Mastermind\csharp\Colors.cs`

```py
namespace Game
{
    /// <summary>
    /// 提供关于可以在代码中使用的颜色信息
    /// </summary>
    public static class Colors
    {
        // 颜色信息数组
        public static readonly ColorInfo[] List = new[]
        {
            // 黑色
            new ColorInfo { ShortName = 'B', LongName = "BLACK"  },
            // 白色
            new ColorInfo { ShortName = 'W', LongName = "WHITE"  },
            // 红色
            new ColorInfo { ShortName = 'R', LongName = "RED"    },
            // 绿色
            new ColorInfo { ShortName = 'G', LongName = "GREEN"  },
            // 橙色
            new ColorInfo { ShortName = 'O', LongName = "ORANGE" },
            // 黄色
            new ColorInfo { ShortName = 'Y', LongName = "YELLOW" },
            // 紫色
            new ColorInfo { ShortName = 'P', LongName = "PURPLE" },
            // 褐色
            new ColorInfo { ShortName = 'T', LongName = "TAN"    }
        };
    }
}
```