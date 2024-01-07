# `basic-computer-games\60_Mastermind\csharp\Colors.cs`

```

// 命名空间 Game
namespace Game
{
    /// <summary>
    /// 提供关于可以在代码中使用的颜色信息
    /// </summary>
    // 静态类 Colors
    public static class Colors
    {
        // 颜色信息数组
        public static readonly ColorInfo[] List = new[]
        {
            // 创建颜色信息对象，包括短名称和长名称
            new ColorInfo { ShortName = 'B', LongName = "BLACK"  },
            new ColorInfo { ShortName = 'W', LongName = "WHITE"  },
            new ColorInfo { ShortName = 'R', LongName = "RED"    },
            new ColorInfo { ShortName = 'G', LongName = "GREEN"  },
            new ColorInfo { ShortName = 'O', LongName = "ORANGE" },
            new ColorInfo { ShortName = 'Y', LongName = "YELLOW" },
            new ColorInfo { ShortName = 'P', LongName = "PURPLE" },
            new ColorInfo { ShortName = 'T', LongName = "TAN"    }
        };
    }
}

```