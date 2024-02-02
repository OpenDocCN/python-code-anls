# `basic-computer-games\87_3-D_Plot\csharp\Program.cs`

```py
// 命名空间声明
namespace Plot
{
    // 类声明
    class Program
    {
        // 主函数
        static void Main(string[] args)
        {
            // 调用打印标题函数
            PrintTitle();

            // 遍历每一行
            foreach (var row in Function.GetRows())
            {
                // 遍历每个元素
                foreach (var z in row)
                {
                    // 调用绘制函数
                    Plot(z);
                }
                // 换行
                Console.WriteLine();
            }
        }

        // 打印标题函数
        private static void PrintTitle()
        {
            // 打印标题
            Console.WriteLine("                                3D Plot");
            Console.WriteLine("               Creative Computing  Morristown, New Jersey");
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
        }

        // 绘制函数
        private static void Plot(int z)
        {
            // 获取当前光标位置的行数
            var x = Console.GetCursorPosition().Top;
            // 设置光标位置
            Console.SetCursorPosition(z, x);
            // 在指定位置打印*
            Console.Write("*");
        }
    }
}
```