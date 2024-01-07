# `basic-computer-games\19_Bunny\csharp\Bunny.cs`

```

// 命名空间声明
namespace Bunny
{
    // 内部类 Bunny
    internal class Bunny
    {
        // 常量声明，ASCII 基数
        private const int asciiBase = 64;
        // 只读整型数组声明，存储一些数据
        private readonly int[] bunnyData = {
            // 数据内容
        };

        // 公共方法 Run
        public void Run()
        {
            // 调用 PrintString 方法，打印字符串 "BUNNY"，并移动到指定位置
            PrintString(33, "BUNNY");
            // 调用 PrintString 方法，打印字符串 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并移动到指定位置
            PrintString(15, "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY");
            // 调用 PrintLines 方法，打印空行，重复 3 次
            PrintLines(3);

            // 实例化 BasicData 对象，传入 bunnyData 数组
            BasicData data = new (bunnyData);

            // 创建字符数组 a，存储前五个数据值对应的字符
            var a = new char[5];
            // 循环读取前五个数据值，并转换为对应的字符存入数组 a
            for (var i = 0; i < 5; ++i)
            {
                a[i] = (char)(asciiBase + data.Read());
            }
            // 调用 PrintLines 方法，打印空行，重复 6 次
            PrintLines(6);

            // 打印空行
            PrintLines(1);
            // 初始化列数为 0
            var col = 0;
            // 循环处理数据
            while (true)
            {
                // 读取下一个数据值
                var x = data.Read();
                // 如果数据值小于 0，开始新的一行
                if (x < 0)
                {
                    // 调用 PrintLines 方法，打印空行
                    PrintLines(1);
                    // 列数归零
                    col = 0;
                    // 继续下一次循环
                    continue;
                }
                // 如果数据值大于 128，结束处理
                if (x > 128) break;
                // 列数增加，移动到指定位置
                col += PrintSpaces(x - col);
                // 读取下一个值
                var y = data.Read();
                // 循环打印字符
                for (var i = x; i <= y; ++i)
                {
                    // 打印字符
                    Console.Write(a[i % 5]);
                    // 列数增加
                    ++col;
                }
            }
            // 调用 PrintLines 方法，打印空行，重复 6 次
            PrintLines(6);
        }
        // 静态方法，打印指定数量的空行
        private static void PrintLines(int count)
        {
            for (var i = 0; i < count; ++i)
                Console.WriteLine();
        }
        // 静态方法，打印指定数量的空格，并返回打印的数量
        private static int PrintSpaces(int count)
        {
            for (var i = 0; i < count; ++i)
                Console.Write(' ');
            return count;
        }
        // 公共静态方法，打印字符串，并移动到指定位置
        public static void PrintString(int tab, string value, bool newLine = true)
        {
            PrintSpaces(tab);
            Console.Write(value);
            if (newLine) Console.WriteLine();
        }

    }
}

```