# `d:/src/tocomm/basic-computer-games\46_Hexapawn\csharp\IReadWriteExtensions.cs`

```
# 导入所需的模块
import System
import Linq
import Games.Common.IO

# 命名空间
namespace Hexapawn;

# 提供模拟 BASIC 解释器键盘输入例程的输入方法
internal static class IReadWriteExtensions
{
    # 获取用户输入的是 Yes 还是 No
    internal static char GetYesNo(this IReadWrite io, string prompt)
    {
        # 循环直到用户输入有效的 Yes 或 No
        while (true)
        {
            # 从输入流中读取用户输入的第一个字符
            var response = io.ReadString($"{prompt} (Y-N)").FirstOrDefault();
            # 如果用户输入的是 Y、y、N 或 n 中的一个
            if ("YyNn".Contains(response))
            {
                # 返回大写形式的用户输入
                return char.ToUpperInvariant(response);
            }
        }
    }
}
    // 定义一个名为ReadMove的静态方法，接受一个IReadWrite类型的参数io和一个字符串类型的参数prompt
    internal static Move ReadMove(this IReadWrite io, string prompt)
    {
        // 创建一个无限循环，直到条件满足才会退出循环
        while(true)
        {
            // 调用io的Read2Numbers方法，传入prompt作为参数，将返回的两个数字分别赋值给from和to
            var (from, to) = io.Read2Numbers(prompt);

            // 调用Move类的TryCreate方法，传入from和to作为参数，如果返回true，则表示成功创建了一个Move对象，将其赋值给move并返回
            if (Move.TryCreate(from, to, out var move))
            {
                return move;
            }

            // 如果TryCreate方法返回false，则输出"Illegal Coordinates."
            io.WriteLine("Illegal Coordinates.");
        }
    }
    }
```

这部分代码是一个缩进错误，应该删除。
```