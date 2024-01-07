# `basic-computer-games\32_Diamond\csharp\StringBuilderExtensions.cs`

```

# 使用 System.Text 命名空间
using System.Text;

# 声明 Diamond 命名空间
namespace Diamond;

# 声明一个内部静态类 StringBuilderExtensions
internal static class StringBuilderExtensions
{
    # 声明一个内部静态方法 PadToLength，接收一个 StringBuilder 对象和一个整数长度参数，返回一个 StringBuilder 对象
    internal static StringBuilder PadToLength(this StringBuilder builder, int length) => 
        # 在 StringBuilder 对象末尾添加空格，使其长度达到指定长度
        builder.Append(' ', length - builder.Length);
}

```