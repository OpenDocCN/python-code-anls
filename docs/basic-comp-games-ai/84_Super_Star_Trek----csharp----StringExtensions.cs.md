# `basic-computer-games\84_Super_Star_Trek\csharp\StringExtensions.cs`

```

# 命名空间 SuperStarTrek 下的内部静态类 StringExtensions
namespace SuperStarTrek;

# 内部静态类 StringExtensions 中的静态方法，用于将单数形式的字符串转换为复数形式
internal static class StringExtensions
{
    # 扩展方法，接收一个字符串和一个整数，根据整数的值判断是否需要加上 "s"，然后返回结果
    internal static string Pluralize(this string singular, int quantity) => singular + (quantity > 1 ? "s" : "");
}

```