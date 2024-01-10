# `basic-computer-games\46_Hexapawn\csharp\IReadWriteExtensions.cs`

```
// 引入命名空间
using System;
using System.Linq;
using Games.Common.IO;

// 声明命名空间 Hexapawn
namespace Hexapawn;

// 提供模拟 BASIC 解释器键盘输入例程的输入方法
internal static class IReadWriteExtensions
{
    // 获取用户输入的是 Yes 还是 No
    internal static char GetYesNo(this IReadWrite io, string prompt)
    {
        // 循环直到用户输入有效的 Yes 或 No
        while (true)
        {
            // 读取用户输入的第一个字符，并转换为大写
            var response = io.ReadString($"{prompt} (Y-N)").FirstOrDefault();
            // 如果用户输入的是 Y、y、N 或 n，则返回对应的大写字符
            if ("YyNn".Contains(response))
            {
                return char.ToUpperInvariant(response);
            }
        }
    }

    // 读取用户输入的移动坐标
    internal static Move ReadMove(this IReadWrite io, string prompt)
    {
        // 循环直到用户输入有效的移动坐标
        while(true)
        {
            // 读取用户输入的两个数字
            var (from, to) = io.Read2Numbers(prompt);

            // 如果输入的坐标可以构成有效的移动，则返回移动对象
            if (Move.TryCreate(from, to, out var move))
            {
                return move;
            }

            // 如果输入的坐标无法构成有效的移动，则提示用户输入非法坐标
            io.WriteLine("Illegal Coordinates.");
        }
    }
}
```