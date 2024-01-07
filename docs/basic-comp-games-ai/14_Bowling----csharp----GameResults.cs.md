# `basic-computer-games\14_Bowling\csharp\GameResults.cs`

```

// 引入命名空间
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// 声明 Bowling 命名空间
namespace Bowling
{
    // 声明 GameResults 类
    public class GameResults
    {
        // 声明 FramesPerGame 常量，表示每局游戏的帧数
        public static readonly int FramesPerGame = 10;
        // 声明 Results 属性，表示每帧的比赛结果
        public FrameResult[] Results { get; set; }

        // 声明 GameResults 类的构造函数
        public GameResults()
        {
            // 初始化 Results 数组，长度为 FramesPerGame
            Results = new FrameResult[FramesPerGame];
            // 遍历 Results 数组，初始化每个元素为新的 FrameResult 对象
            for (int i = 0; i < FramesPerGame; ++i)
            {
                Results[i] = new FrameResult();
            }
        }
    }
}

```