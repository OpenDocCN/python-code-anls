# `basic-computer-games\38_Fur_Trader\csharp\Program.cs`

```

// 引入 System 命名空间
using System;

// 声明 FurTrader 命名空间
namespace FurTrader
{
    // 声明 Program 类
    public class Program
    {
        /// <summary>
        /// 当应用程序启动时，此函数将自动被调用
        /// </summary>
        /// <param name="args">命令行参数</param>
        public static void Main(string[] args)
        {
            // 创建主游戏类的实例
            var game = new Game();

            // 调用其 GameLoop 函数。这将在循环中无限地播放游戏，直到玩家选择退出
            game.GameLoop();
        }
    }
}

```