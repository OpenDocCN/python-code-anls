# `69_Pizza\csharp\StringBuilderExtensions.cs`

```
// 命名空间 Pizza
namespace Pizza
{
    // 内部静态类 StringBuilderExtensions
    internal static class StringBuilderExtensions
    {
        /// <summary>
        /// 用于添加特定值的新行的扩展方法。
        /// </summary>
        /// <param name="stringBuilder">被扩展的类。</param>
        /// <param name="value">将被重复的值。</param>
        /// <param name="numberOfLines">将被添加的行数。</param>
        public static void AppendLine(this StringBuilder stringBuilder, string value, int numberOfLines)
        {
            // 循环添加指定行数的值
            for (int i = 0; i < numberOfLines; i++)
            {
                stringBuilder.AppendLine(value);
            }
        }
    }
}
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源，避免内存泄漏。
```