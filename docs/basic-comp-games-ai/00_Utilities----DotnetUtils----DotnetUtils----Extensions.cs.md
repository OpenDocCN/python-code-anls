# `basic-computer-games\00_Utilities\DotnetUtils\DotnetUtils\Extensions.cs`

```py
// 命名空间 DotnetUtils
namespace DotnetUtils;

// 定义静态类 Extensions
public static class Extensions {
    // 定义 SelectT 方法，接受一个元组和一个选择器函数，返回选择器函数作用后的结果集
    public static IEnumerable<TResult> SelectT<T1, T2, TResult>(this IEnumerable<(T1, T2)> src, Func<T1, T2, TResult> selector) =>
        src.Select(x => selector(x.Item1, x.Item2));
    // 定义 SelectT 方法，接受一个三元组和一个选择器函数，返回选择器函数作用后的结果集
    public static IEnumerable<TResult> SelectT<T1, T2, T3, TResult>(this IEnumerable<(T1, T2, T3)> src, Func<T1, T2, T3, TResult> selector) =>
        src.Select(x => selector(x.Item1, x.Item2, x.Item3));
    // 定义 WithIndex 方法，为元组添加索引，返回带索引的元组集合
    public static IEnumerable<(T1, T2, int)> WithIndex<T1, T2>(this IEnumerable<(T1, T2)> src) => src.Select((x, index) => (x.Item1, x.Item2, index));

    // 定义 None 方法，判断集合是否为空或者不满足给定条件
    public static bool None<T>(this IEnumerable<T> src, Func<T, bool>? predicate = null) =>
        predicate is null ?
            !src.Any() :
            !src.Any(predicate);

    // 定义 IsNullOrWhitespace 方法，判断字符串是否为空或者只包含空白字符
    public static bool IsNullOrWhitespace([NotNullWhen(false)] this string? s) => string.IsNullOrWhiteSpace(s);

    // 定义 RelativePath 方法，返回相对路径
    [return: NotNullIfNotNull("path")]
    public static string? RelativePath(this string? path, string? rootPath) {
        // 如果路径为空或者只包含空白字符，直接返回路径
        if (
            path.IsNullOrWhitespace() ||
            rootPath.IsNullOrWhitespace()
        ) { return path; }

        // 移除路径末尾的反斜杠
        path = path.TrimEnd('\\'); // remove trailing backslash, if present
        // 返回相对路径
        return GetRelativePath(rootPath, path.TrimEnd('\\'));
    }
}
```