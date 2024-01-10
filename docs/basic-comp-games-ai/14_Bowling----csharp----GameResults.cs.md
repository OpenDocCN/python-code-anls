# `basic-computer-games\14_Bowling\csharp\GameResults.cs`

```
// 引入命名空间
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// 命名空间 Bowling
namespace Bowling
{
    // 定义公共类 GameResults
    public class GameResults
    {
        // 定义静态只读整型变量 FramesPerGame，表示每局游戏的帧数为10
        public static readonly int FramesPerGame = 10;
        // 定义属性 Results，表示每帧的比赛结果
        public FrameResult[] Results { get; set; }

        // 构造函数，初始化 Results 数组，每个元素为一个 FrameResult 对象
        public GameResults()
        {
            // 初始化 Results 数组，长度为 FramesPerGame
            Results = new FrameResult[FramesPerGame];
            // 遍历 Results 数组，为每个元素赋值为一个新的 FrameResult 对象
            for (int i = 0; i < FramesPerGame; ++i)
            {
                Results[i] = new FrameResult();
            }
        }
    }
}
```