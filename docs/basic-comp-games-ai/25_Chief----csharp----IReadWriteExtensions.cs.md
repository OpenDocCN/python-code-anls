# `basic-computer-games\25_Chief\csharp\IReadWriteExtensions.cs`

```

# 命名空间 Chief，包含了 IReadWriteExtensions 类
namespace Chief;

# IReadWriteExtensions 类是一个内部静态类
internal static class IReadWriteExtensions
{
    # 读取 Yes 的扩展方法，接受格式化字符串和数字值作为参数
    internal static bool ReadYes(this IReadWrite io, string format, Number value) =>
        io.ReadYes(string.Format(format, value));
    # 读取 Yes 的扩展方法，接受提示字符串作为参数
    internal static bool ReadYes(this IReadWrite io, string prompt) =>
        io.ReadString(prompt).Equals("Yes", StringComparison.InvariantCultureIgnoreCase);
}

```