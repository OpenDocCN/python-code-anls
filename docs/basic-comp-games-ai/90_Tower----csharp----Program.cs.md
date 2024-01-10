# `basic-computer-games\90_Tower\csharp\Program.cs`

```
using System;  // 导入系统命名空间
using Tower.Resources;  // 导入 Tower.Resources 命名空间
using Tower.UI;  // 导入 Tower.UI 命名空间

namespace Tower  // 定义 Tower 命名空间
{
    class Program  // 定义 Program 类
    {
        static void Main(string[] args)  // 定义程序入口方法
        {
            Console.Write(Strings.Title);  // 在控制台输出标题字符串

            do  // 开始 do-while 循环
            {
                Console.Write(Strings.Intro);  // 在控制台输出介绍字符串

                if (!Input.TryReadNumber(Prompt.DiskCount, out var diskCount)) { return; }  // 尝试读取输入的数字，如果失败则返回

                var game = new Game(diskCount);  // 创建一个新的游戏对象

                if (!game.Play()) { return; }  // 如果游戏无法进行，则返回

            } while (Input.ReadYesNo(Strings.PlayAgainPrompt, Strings.YesNoPrompt));  // 循环直到输入不再是肯定的

            Console.Write(Strings.Thanks);  // 在控制台输出感谢字符串
        }
    }
}
```