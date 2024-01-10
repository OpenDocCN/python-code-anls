# `basic-computer-games\58_Love\csharp\StringBuilderExtensions.cs`

```
# 使用 System.Text 命名空间
using System.Text;

# 声明 Love 命名空间
namespace Love;

# 声明内部静态类 StringBuilderExtensions
internal static class StringBuilderExtensions
{
    # 声明内部静态方法 AppendLines，接受 StringBuilder 对象和行数作为参数
    internal static StringBuilder AppendLines(this StringBuilder builder, int count)
    {
        # 使用 for 循环迭代行数次
        for (int i = 0; i < count; i++)
        {
            # 在 StringBuilder 对象中追加换行符
            builder.AppendLine();
        }

        # 返回追加了换行符的 StringBuilder 对象
        return builder;
    }
}
```