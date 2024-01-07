# `basic-computer-games\90_Tower\csharp\Program.cs`

```

// 引入命名空间
using System;
using Tower.Resources;
using Tower.UI;

// 定义程序类
namespace Tower
{
    class Program
    {
        // 主函数
        static void Main(string[] args)
        {
            // 输出标题字符串
            Console.Write(Strings.Title);

            // 循环进行游戏
            do
            {
                // 输出游戏介绍字符串
                Console.Write(Strings.Intro);

                // 尝试读取输入的数字作为盘子数量
                if (!Input.TryReadNumber(Prompt.DiskCount, out var diskCount)) { return; }

                // 创建游戏对象
                var game = new Game(diskCount);

                // 进行游戏，如果游戏结束则退出循环
                if (!game.Play()) { return; }
            } while (Input.ReadYesNo(Strings.PlayAgainPrompt, Strings.YesNoPrompt)); // 循环条件为是否继续游戏

            // 输出感谢字符串
            Console.Write(Strings.Thanks);
        }
    }
}

```