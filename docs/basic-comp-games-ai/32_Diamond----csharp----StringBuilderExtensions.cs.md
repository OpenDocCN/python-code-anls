# `32_Diamond\csharp\StringBuilderExtensions.cs`

```
# 使用 System.Text 命名空间
using System.Text;

# 在 Diamond 命名空间下创建一个内部的静态类 StringBuilderExtensions
namespace Diamond;

internal static class StringBuilderExtensions
{
    # 创建一个内部的静态方法 PadToLength，接受一个 StringBuilder 对象和一个整数长度作为参数，返回一个 StringBuilder 对象
    internal static StringBuilder PadToLength(this StringBuilder builder, int length) => 
        # 在原 StringBuilder 对象的末尾添加空格，直到达到指定的长度
        builder.Append(' ', length - builder.Length);
}
```