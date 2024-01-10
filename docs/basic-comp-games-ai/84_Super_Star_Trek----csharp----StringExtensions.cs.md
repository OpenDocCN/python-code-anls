# `basic-computer-games\84_Super_Star_Trek\csharp\StringExtensions.cs`

```
# 创建名为 SuperStarTrek 的命名空间，并定义一个内部静态类 StringExtensions
namespace SuperStarTrek;

# 在 StringExtensions 类中定义一个内部静态方法 Pluralize，用于将单数形式的字符串转换为复数形式
internal static class StringExtensions
{
    # 使用扩展方法的方式，接收一个字符串和一个整数参数，根据数量判断是否需要加上 "s" 后缀
    internal static string Pluralize(this string singular, int quantity) => singular + (quantity > 1 ? "s" : "");
}
```