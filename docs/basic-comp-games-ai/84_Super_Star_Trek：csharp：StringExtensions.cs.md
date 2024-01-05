# `d:/src/tocomm/basic-computer-games\84_Super_Star_Trek\csharp\StringExtensions.cs`

```
# 定义命名空间 SuperStarTrek
namespace SuperStarTrek;

# 定义一个静态类 StringExtensions
internal static class StringExtensions
{
    # 定义一个静态方法 Pluralize，接收一个字符串和一个整数参数，返回一个字符串
    internal static string Pluralize(this string singular, int quantity) => singular + (quantity > 1 ? "s" : "");
}
```