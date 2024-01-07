# `basic-computer-games\07_Basketball\csharp\IReadWriteExtensions.cs`

```

// 使用 Games.Common.IO 命名空间
using Games.Common.IO;

// 声明 Basketball 命名空间
namespace Basketball;

// 声明 IReadWriteExtensions 类，为静态类
internal static class IReadWriteExtensions
{
    // 为 IReadWrite 接口添加 ReadDefense 方法，返回浮点数
    public static float ReadDefense(this IReadWrite io, string prompt)
    {
        // 循环直到条件满足
        while (true)
        {
            // 调用 IReadWrite 接口的 ReadNumber 方法，传入提示信息
            var defense = io.ReadNumber(prompt);
            // 如果防守值大于等于 6，则返回防守值
            if (defense >= 6) { return defense; }
        }
    }

    // 为 IReadWrite 接口添加 TryReadInteger 方法，返回布尔值和整数值
    private static bool TryReadInteger(this IReadWrite io, string prompt, out int intValue)
    {
        // 调用 IReadWrite 接口的 ReadNumber 方法，传入提示信息
        var floatValue = io.ReadNumber(prompt);
        // 将浮点数转换为整数
        intValue = (int)floatValue;
        // 检查转换后的整数是否与原始浮点数相等，返回结果
        return intValue == floatValue;
    }

    // 为 IReadWrite 接口添加 ReadShot 方法，返回 Shot 枚举类型的可空值
    public static Shot? ReadShot(this IReadWrite io, string prompt)
    {
        // 循环直到条件满足
        while (true)
        {
            // 调用 TryReadInteger 方法，并尝试将结果转换为 Shot 枚举类型
            if (io.TryReadInteger(prompt, out var value) && Shot.TryGet(value, out var shot))
            {
                // 如果转换成功，则返回 Shot 枚举类型的值
                return shot;
            }
            // 如果转换失败，则输出错误信息，继续循环
            io.Write("Incorrect answer.  Retype it. ");
        }
    }
}

```