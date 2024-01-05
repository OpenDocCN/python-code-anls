# `d:/src/tocomm/basic-computer-games\25_Chief\csharp\IReadWriteExtensions.cs`

```
namespace Chief;  # 命名空间声明，定义了代码的作用域

internal static class IReadWriteExtensions  # 定义了一个静态类 IReadWriteExtensions，用于扩展 IReadWrite 接口的功能

{
    internal static bool ReadYes(this IReadWrite io, string format, Number value) =>  # 定义了一个扩展方法 ReadYes，接受一个 IReadWrite 对象、一个格式化字符串和一个 Number 值作为参数，并返回一个布尔值
        io.ReadYes(string.Format(format, value));  # 调用了另一个重载的 ReadYes 方法，传入格式化后的字符串作为参数

    internal static bool ReadYes(this IReadWrite io, string prompt) =>  # 定义了另一个扩展方法 ReadYes，接受一个 IReadWrite 对象和一个提示字符串作为参数，并返回一个布尔值
        io.ReadString(prompt).Equals("Yes", StringComparison.InvariantCultureIgnoreCase);  # 调用了 IReadWrite 接口的 ReadString 方法，然后使用 Equals 方法比较返回的字符串和 "Yes"，并忽略大小写

}
```