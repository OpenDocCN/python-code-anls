# `basic-computer-games\07_Basketball\csharp\IReadWriteExtensions.cs`

```py
// 使用 Games.Common.IO 命名空间
using Games.Common.IO;

// 声明 Basketball 命名空间
namespace Basketball
{
    // 声明 IReadWriteExtensions 静态类
    internal static class IReadWriteExtensions
    {
        // 为 IReadWrite 接口添加 ReadDefense 方法，用于读取防守值
        public static float ReadDefense(this IReadWrite io, string prompt)
        {
            // 循环读取防守值
            while (true)
            {
                // 调用 ReadNumber 方法读取防守值
                var defense = io.ReadNumber(prompt);
                // 如果防守值大于等于 6，则返回防守值
                if (defense >= 6) { return defense; }
            }
        }

        // 为 IReadWrite 接口添加 TryReadInteger 方法，尝试读取整数值
        private static bool TryReadInteger(this IReadWrite io, string prompt, out int intValue)
        {
            // 调用 ReadNumber 方法读取数值
            var floatValue = io.ReadNumber(prompt);
            // 将浮点数值转换为整数值
            intValue = (int)floatValue;
            // 判断转换后的整数值是否与原始浮点数值相等
            return intValue == floatValue;
        }

        // 为 IReadWrite 接口添加 ReadShot 方法，用于读取投篮动作
        public static Shot? ReadShot(this IReadWrite io, string prompt)
        {
            // 循环读取投篮动作
            while (true)
            {
                // 尝试读取整数值，并根据整数值获取对应的投篮动作
                if (io.TryReadInteger(prompt, out var value) && Shot.TryGet(value, out var shot))
                {
                    return shot;
                }
                // 如果无法获取有效的投篮动作，则提示重新输入
                io.Write("Incorrect answer.  Retype it. ");
            }
        }
    }
}
```