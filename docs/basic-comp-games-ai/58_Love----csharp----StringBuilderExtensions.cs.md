# `basic-computer-games\58_Love\csharp\StringBuilderExtensions.cs`

```

# 引入 System.Text 命名空间
using System.Text;

# 声明名为 Love 的命名空间
namespace Love;

# 声明名为 StringBuilderExtensions 的静态类
internal static class StringBuilderExtensions
{
    # 声明名为 AppendLines 的静态方法，接受 StringBuilder 对象和行数作为参数
    internal static StringBuilder AppendLines(this StringBuilder builder, int count)
    {
        # 循环 count 次，每次在 StringBuilder 对象中追加一个换行符
        for (int i = 0; i < count; i++)
        {
            builder.AppendLine();
        }

        # 返回追加了换行符后的 StringBuilder 对象
        return builder;
    }
}

```