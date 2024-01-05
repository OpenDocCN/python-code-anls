# `d:/src/tocomm/basic-computer-games\07_Basketball\csharp\IReadWriteExtensions.cs`

```
using Games.Common.IO;  // 导入 Games.Common.IO 命名空间

namespace Basketball;  // 定义 Basketball 命名空间

internal static class IReadWriteExtensions  // 定义 IReadWriteExtensions 静态类
{
    public static float ReadDefense(this IReadWrite io, string prompt)  // 定义 ReadDefense 方法，接受 IReadWrite 和 prompt 作为参数，返回 float 类型
    {
        while (true)  // 进入无限循环
        {
            var defense = io.ReadNumber(prompt);  // 调用 io 的 ReadNumber 方法，将结果赋值给 defense
            if (defense >= 6) { return defense; }  // 如果 defense 大于等于 6，则返回 defense
        }
    }

    private static bool TryReadInteger(this IReadWrite io, string prompt, out int intValue)  // 定义 TryReadInteger 方法，接受 IReadWrite、prompt 和 intValue 作为参数，返回 bool 类型
    {
        var floatValue = io.ReadNumber(prompt);  // 调用 io 的 ReadNumber 方法，将结果赋值给 floatValue
        intValue = (int)floatValue;  // 将 floatValue 转换为 int 类型，赋值给 intValue
        return intValue == floatValue;  // 返回 intValue 是否等于 floatValue
    }
}
    }  # 结束 ReadShot 方法的定义

    public static Shot? ReadShot(this IReadWrite io, string prompt)  # 定义一个名为 ReadShot 的扩展方法，接受一个 IReadWrite 类型的参数和一个字符串参数
    {
        while (true)  # 进入一个无限循环
        {
            if (io.TryReadInteger(prompt, out var value) && Shot.TryGet(value, out var shot))  # 如果尝试从输入流中读取一个整数，并且能够将其转换为 Shot 对象
            {
                return shot;  # 返回转换后的 Shot 对象
            }
            io.Write("Incorrect answer.  Retype it. ");  # 如果无法成功转换，提示用户重新输入
        }
    }
}
```