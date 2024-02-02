# `basic-computer-games\25_Chief\csharp\IReadWriteExtensions.cs`

```py
# 定义了一个命名空间 Chief，包含了一个内部静态类 IReadWriteExtensions
namespace Chief;

internal static class IReadWriteExtensions
{
    # 定义了一个内部静态方法 ReadYes，接收一个 IReadWrite 类型的参数 io，一个格式字符串 format 和一个 Number 类型的参数 value，返回一个布尔值
    internal static bool ReadYes(this IReadWrite io, string format, Number value) =>
        # 调用 ReadYes 方法，传入格式化后的字符串和 value 参数
        io.ReadYes(string.Format(format, value));
    # 定义了一个内部静态方法 ReadYes，接收一个 IReadWrite 类型的参数 io 和一个字符串 prompt，返回一个布尔值
    internal static bool ReadYes(this IReadWrite io, string prompt) =>
        # 调用 ReadString 方法，传入 prompt 参数，然后将结果与 "Yes" 进行不区分大小写的比较
        io.ReadString(prompt).Equals("Yes", StringComparison.InvariantCultureIgnoreCase);
}
```