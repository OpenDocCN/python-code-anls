# `basic-computer-games\87_3-D_Plot\csharp\Program.cs`

```

// 引入 System 命名空间
using System;

// 定义 Plot 类
namespace Plot
{
    // 定义 Program 类
    class Program
    {
        // 程序入口
        static void Main(string[] args)
        {
            // 调用 PrintTitle 方法打印标题
            PrintTitle();

            // 遍历 Function.GetRows() 返回的行
            foreach (var row in Function.GetRows())
            {
                // 遍历行中的元素
                foreach (var z in row)
                {
                    // 调用 Plot 方法绘制图形
                    Plot(z);
                }
                // 换行
                Console.WriteLine();
            }
        }

        // 打印标题
        private static void PrintTitle()
        {
            // 打印标题信息
            Console.WriteLine("                                3D Plot");
            Console.WriteLine("               Creative Computing  Morristown, New Jersey");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
        }

        // 绘制图形
        private static void Plot(int z)
        {
            // 获取当前光标位置的行数
            var x = Console.GetCursorPosition().Top;
            // 设置光标位置并打印 *
            Console.SetCursorPosition(z, x);
            Console.Write("*");
        }
    }
}

```