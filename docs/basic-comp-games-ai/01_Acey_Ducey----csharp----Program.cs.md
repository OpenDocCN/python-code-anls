# `basic-computer-games\01_Acey_Ducey\csharp\Program.cs`

```

// 引入命名空间
using System;
using System.Threading;

// 定义应用程序的入口点
namespace AceyDucey
{
    /// <summary>
    /// 应用程序的入口点
    /// </summary>
    class Program
    {
        /// <summary>
        /// 当应用程序启动时，将自动调用此函数
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {
            // 创建主游戏类的实例
            Game game = new Game();

            // 调用其 GameLoop 函数。这将在循环中无休止地播放游戏，直到玩家选择退出。
            game.GameLoop();
        }
    }
}

```