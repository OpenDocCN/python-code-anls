# `basic-computer-games\46_Hexapawn\csharp\IReadWriteExtensions.cs`

```

// 引入命名空间
using System;
using System.Linq;
using Games.Common.IO;

// 创建名为 IReadWriteExtensions 的静态类，提供模拟 BASIC 解释器键盘输入例程的输入方法
internal static class IReadWriteExtensions
{
    // 创建名为 GetYesNo 的扩展方法，接收 IReadWrite 对象和提示信息作为参数，返回一个字符
    internal static char GetYesNo(this IReadWrite io, string prompt)
    {
        // 循环直到条件满足
        while (true)
        {
            // 从输入中获取第一个字符
            var response = io.ReadString($"{prompt} (Y-N)").FirstOrDefault();
            // 如果输入的字符是 Y、y、N 或 n 中的一个，则返回大写形式的该字符
            if ("YyNn".Contains(response))
            {
                return char.ToUpperInvariant(response);
            }
        }
    }

    // 创建名为 ReadMove 的扩展方法，接收 IReadWrite 对象和提示信息作为参数，返回一个 Move 对象
    internal static Move ReadMove(this IReadWrite io, string prompt)
    {
        // 循环直到条件满足
        while(true)
        {
            // 从输入中获取两个数字
            var (from, to) = io.Read2Numbers(prompt);

            // 如果可以创建移动对象，则返回该对象
            if (Move.TryCreate(from, to, out var move))
            {
                return move;
            }

            // 输出错误信息
            io.WriteLine("Illegal Coordinates.");
        }
    }
}

```