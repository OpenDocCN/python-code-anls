# `basic-computer-games\86_Target\csharp\Program.cs`

```
// 引入命名空间
using System;
using System.Reflection;
using Games.Common.IO;
using Games.Common.Randomness;

// 定义命名空间和类
namespace Target
{
    class Program
    {
        // 程序入口
        static void Main()
        {
            // 创建控制台输入输出对象
            var io = new ConsoleIO();
            // 创建游戏对象，传入输入输出对象和随机数生成器
            var game = new Game(io, new FiringRange(new RandomNumberGenerator()));

            // 开始游戏
            Play(game, io, () => true);
        }

        // 游戏进行函数，接受游戏对象、输入输出对象和再玩一次的函数
        public static void Play(Game game, TextIO io, Func<bool> playAgain)
        {
            // 显示游戏标题和说明
            DisplayTitleAndInstructions(io);

            // 当再玩一次的函数返回 true 时，循环进行游戏
            while (playAgain())
            {
                // 进行游戏
                game.Play();

                // 输出空行和提示信息
                io.WriteLine();
                io.WriteLine();
                io.WriteLine();
                io.WriteLine();
                io.WriteLine();
                io.WriteLine("Next target...");
                io.WriteLine();
            }
        }

        // 显示游戏标题和说明
        private static void DisplayTitleAndInstructions(TextIO io)
        {
            // 使用程序集中的资源文件流来显示标题和说明
            using var stream = Assembly.GetExecutingAssembly()
                .GetManifestResourceStream("Target.Strings.TitleAndInstructions.txt");
            io.Write(stream);
        }
    }
}
```